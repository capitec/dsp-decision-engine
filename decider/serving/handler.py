from __future__ import annotations

import inspect
import os
import pkgutil
import sys
import threading
import typing as t
from dataclasses import dataclass, field

import polars as pl
from pydantic_core import to_json

import decider.exceptions as exc
from decider.config import ConfigStore, Version
from decider.engine import Engine
from decider.engine.ir.decls import base_annotation
from decider.engine.run import Executable
from decider.steps import ConfigurableStep, Step
from .format import DEFAULT_OUTPUT_FORMATTERS, Response
from .media_types import MediaType
from .parse import DEFAULT_INPUT_HANDLERS


class Live(t.NamedTuple):
    """A built config version: what one request scores against."""

    version: Version
    executable: Executable
    params: dict[str, t.Any] | None


@dataclass
class RequestHandler:
    """Serves a pipeline from code with params and `ConfigurableStep` documents from a config store.

    `pipeline` is a `Step`, or a function returning one, or the import path of
    either (`"myapp.pipeline:build"`). Each argument of the function names a
    key of the config version; that document is loaded with
    `ConfigurableStep.load` and passed in. The version's `"params"` document
    is the params document every request runs with.

    Config changes are explicit: `stage(version)` builds and warms a version
    off the request path, `activate()` swaps it in atomically, `rollback()`
    goes back to the previous one. Errors propagate to the caller, and the
    active version keeps serving. `POST /invocations` with a JSON object
    scores one record; a JSON array, JSONL, CSV or Parquet body runs as a frame.

    Override any `*_fn` method in a `Handler` subclass in `inference.py` to
    change how requests are parsed, scored or formatted.

    Example::

        # myapp/pipeline.py
        def build(tree: ConfigurableStep) -> Step:      # the version's "tree" document
            return flow(prepare, tree, decide, name="credit")

        store = JsonFileStore(basepath="configs")
        store.create_version({"tree": tree_doc, "params": {"credit": {"decide": {"cut": 0.7}}}})

        handler = RequestHandler(store, "myapp.pipeline:build")
        handler.stage()          # latest version; or stage("1.2.0")
        handler.activate()
        handler.rollback()       # back to the previously active version

        # or serve it: DECIDER_API__PIPELINE=myapp.pipeline:build, then
        # uvicorn --factory decider.serving.servers.starlette:create_app
    """

    store: ConfigStore
    pipeline: t.Any = "pipeline:build"
    mode: str = "fused"
    _active: Live | None = field(default=None, init=False, repr=False)
    _staged: Live | None = field(default=None, init=False, repr=False)
    _history: list[Live] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def active(self) -> Version | None:
        """The version requests are scored against, once one is activated."""
        live = self._active
        return live.version if live is not None else None

    @property
    def staged(self) -> Version | None:
        """The version `activate()` would switch to."""
        live = self._staged
        return live.version if live is not None else None

    def stage(self, version: str | Version | None = None) -> Version:
        """Build and warm `version` (default: the store's latest) without serving it."""
        with self._lock:
            if version is None:
                version = self.store.latest_version()
                if version is None:
                    raise RuntimeError("the config store has no versions to stage.")
            versioned = self.store.read(version)
            exe = Engine().bind(self.pipeline_fn(versioned.config), mode=self.mode)
            params = versioned.config.get("params")
            _warm(exe, params)
            self._staged = Live(versioned.version, exe, params)
            return versioned.version

    def activate(self) -> Version:
        """Serve the staged version; the one it replaces is kept for `rollback()`."""
        with self._lock:
            if self._staged is None:
                raise RuntimeError("nothing staged; call stage() before activate().")
            if self._active is not None:
                # ponytail: history is unbounded; cap it if versions are activated often in one process.
                self._history.append(self._active)
            self._active, self._staged = self._staged, None
            return self._active.version

    def rollback(self) -> Version:
        """Serve the previously active version again."""
        with self._lock:
            if not self._history:
                raise RuntimeError("no previously active version to roll back to.")
            self._active = self._history.pop()
            return self._active.version

    async def init_fn(self):
        self.stage()
        self.activate()

    def pipeline_fn(self, config: dict[str, t.Any]) -> Step:
        target = pkgutil.resolve_name(self.pipeline) if isinstance(self.pipeline, str) else self.pipeline
        if isinstance(target, Step):
            return target
        docs = {}
        for name, p in inspect.signature(target).parameters.items():
            if name in config:
                docs[name] = ConfigurableStep.load(config[name])
            elif p.default is p.empty:
                raise KeyError(f"the config version has no {name!r} document for argument {name!r} of {self.pipeline!r}")
        return target(**docs)

    def module_fn(self) -> Live:
        # Read once per request, so one request never mixes two versions.
        live = self._active
        if live is None:
            raise RuntimeError("no config version is active; call stage() then activate().")
        return live

    async def input_fn(self, data: bytes, content_type: str) -> dict[str, t.Any] | pl.DataFrame:
        handler = DEFAULT_INPUT_HANDLERS.get(content_type)
        if handler is None:
            raise exc.UnsupportedContentTypeError(f"Unsupported content type: {content_type!r}")
        try:
            return handler(data)
        except Exception as e:
            raise exc.InputParsingError(str(e))

    def output_fn(self, output: dict[str, t.Any] | pl.DataFrame, accept: str) -> Response:
        if isinstance(output, dict):
            if accept in (MediaType.ANY.value, MediaType.APPLICATION_JSON.value):
                return Response(to_json(output), MediaType.APPLICATION_JSON.value)
            output = pl.DataFrame([output])
        formatter = DEFAULT_OUTPUT_FORMATTERS.get(accept)
        if formatter is None:
            raise exc.UnsupportedAcceptError(f"Unsupported Accept type: {accept!r}")
        try:
            response = formatter(output)
            if response.media_type is None:
                if accept == MediaType.ANY.value:
                    raise exc.DeciderRuntimeError("Configured Format for MediaType.ANY must return a Response with a specific media_type, got None")
                response = Response(content=response.content, media_type=accept)
            return response
        except exc.DeciderError:
            raise
        except Exception as e:
            raise exc.OutputFormattingError(str(e))

    async def process_fn(self, data: bytes, accept: str, content_type: str) -> Response:
        live = self.module_fn()
        request = await self.input_fn(data, content_type)
        if isinstance(request, dict):
            result = live.executable.score(request, live.params)
        else:
            result = live.executable.run(request, live.params)
        return self.output_fn(result, accept)

    async def shutdown_fn(self):
        pass


_DUMMY = {bool: False, int: 1, str: ""}


def _warm(exe: Executable, params: t.Any) -> None:
    # One arbitrary row through both paths compiles every kernel before a request needs it;
    # 1 rather than 0 so an ordinary ratio doesn't divide by zero.
    record = {v.name: _DUMMY.get(base_annotation(v.annotation), 1.0) for v in exe.plan.versions if v.producer is None}
    exe.score(record, params)
    exe.run(pl.DataFrame([record]), params)


def construct_handler_from_settings() -> RequestHandler:
    """The handler `settings.api` describes: `Handler` from `inference.py` in `code_path` if there is one."""
    from decider.settings import settings

    code_path = os.path.abspath(settings.api.code_path)
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    module_name, _, class_name = settings.api.handler.partition(":")
    constructor = RequestHandler
    if os.path.exists(os.path.join(code_path, f"{module_name}.py")):
        constructor = getattr(pkgutil.resolve_name(module_name), class_name, RequestHandler)
    return constructor(settings.config.get(), settings.api.pipeline, settings.api.mode)
