from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
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
from decider.engine.run import Executable
from decider.steps import ConfigurableStep, Step
from .format import DEFAULT_OUTPUT_FORMATTERS, Response
from .parse import DEFAULT_INPUT_HANDLERS, coerce_frame, coerce_record, dummy, has_date


class Live(t.NamedTuple):
    """A built config version: what one request scores against."""

    version: Version
    executable: Executable
    params: dict[str, t.Any] | None
    dates: dict[str, t.Any] = {}


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
    change how requests are parsed, scored or formatted. JSON dates arrive as
    ISO strings and are converted to the pipeline's declared `date` and
    `datetime` inputs, including inside `list[date]`, `date | None` and
    TypedDict fields (`accounts: list[Account]`); a bare `dict` is left as sent.

    `stage` warms a version by scoring `sample_request` (a JSON file, by
    default `sample_request.json` in `code_path`) if it exists, else a
    synthetic record; override `warm_fn` to warm another way.

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
    sample_request: str | None = None
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
        """Build and warm `version` (default: the store's latest) without serving it.

        Example::

            handler.stage("1.3.0")
            handler.activate()
        """
        with self._lock:
            if version is None:
                version = self.store.latest_version()
                if version is None:
                    raise RuntimeError("the config store has no versions to stage.")
            versioned = self.store.read(version)
            exe = Engine().bind(self.pipeline_fn(versioned.config), mode=self.mode)
            params = versioned.config.get("params")
            self.warm_fn(exe, params)
            self._staged = Live(versioned.version, exe, params, _dates(exe))
            return versioned.version

    def activate(self) -> Version:
        """Serve the staged version; the one it replaces is kept for `rollback()`.

        Example::

            handler.stage()
            handler.activate()   # Version(1, 3, 0), now answering requests
        """
        with self._lock:
            if self._staged is None:
                raise RuntimeError("nothing staged; call stage() before activate().")
            if self._active is not None:
                # ponytail: history is unbounded; cap it if versions are activated often in one process.
                self._history.append(self._active)
            self._active, self._staged = self._staged, None
            return self._active.version

    def rollback(self) -> Version:
        """Serve the previously active version again.

        Example::

            handler.rollback()   # Version(1, 2, 0)
        """
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
            if accept in ("*/*", "application/json"):
                return Response(to_json(output), "application/json")
            output = pl.DataFrame([output])
        formatter = DEFAULT_OUTPUT_FORMATTERS.get(accept)
        if formatter is None:
            raise exc.UnsupportedAcceptError(f"Unsupported Accept type: {accept!r}")
        media_type, write = formatter
        try:
            return Response(write(output), media_type)
        except Exception as e:
            raise exc.OutputFormattingError(str(e))

    async def process_fn(self, data: bytes, accept: str, content_type: str) -> Response:
        live = self.module_fn()
        request = await self.input_fn(data, content_type)
        if isinstance(request, dict):
            result = live.executable.score(coerce_record(request, live.dates), live.params)
        else:
            result = live.executable.run(coerce_frame(request, live.dates), live.params)
        return self.output_fn(result, accept)

    def warm_fn(self, executable: Executable, params: t.Any) -> None:
        # One record through both paths compiles every kernel before a request needs it.
        sample = self.sample_request if self.sample_request and os.path.exists(self.sample_request) else None
        if sample:
            hint = (f"Fix {sample} so it is a request this pipeline answers. Dates inside a bare `dict` input "
                    f"stay strings; annotate it as a TypedDict with `date` fields to have them converted.")
        else:
            record = {v.name: dummy(v.annotation) for v in _inputs(executable)}
            where = self.sample_request or "a JSON file passed as RequestHandler(sample_request=...)"
            hint = (f"The synthetic inputs {record!r} don't suit this pipeline; put a representative request "
                    f"in {where} and staging warms with it instead.")
        try:
            if sample:
                with open(sample) as f:
                    record = coerce_record(json.load(f), _dates(executable))
            executable.score(record, params)
            executable.run(pl.DataFrame([record]), params)
        except exc.ParamsError:
            raise
        except Exception as e:
            raise exc.DeciderError(f"warm-up failed with {type(e).__name__}: {e}. {hint}") from e

    async def shutdown_fn(self):
        pass


def _inputs(exe: Executable) -> list:
    return [v for v in exe.plan.versions if v.producer is None]


def _dates(exe: Executable) -> dict[str, t.Any]:
    return {v.name: v.annotation for v in _inputs(exe) if has_date(v.annotation)}


def construct_handler_from_settings() -> RequestHandler:
    """The handler `settings.api` describes: `Handler` from `inference.py` in `code_path` if there is one."""
    from decider.settings import settings

    code_path = os.path.abspath(settings.api.code_path)
    # First, even when PYTHONPATH already lists it behind another project with its own pipeline.py.
    if code_path in sys.path:
        sys.path.remove(code_path)
    sys.path.insert(0, code_path)
    pipeline_module = importlib.import_module(settings.api.pipeline.partition(":")[0])
    # Shows a pipeline imported from the wrong project before it serves the wrong answers.
    print(f"decider: pipeline {settings.api.pipeline} from {pipeline_module.__file__}", file=sys.stderr)
    module_name, _, class_name = settings.api.handler.partition(":")
    constructor = RequestHandler
    try:
        found = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:  # a missing parent package of a dotted name
        found = False
    if found:
        handler_module = importlib.import_module(module_name)
        constructor = getattr(handler_module, class_name, None)
        if constructor is None:
            raise AttributeError(
                f"{handler_module.__file__} has no {class_name!r}; define `class {class_name}(RequestHandler)` "
                f"there, or point DECIDER_API__HANDLER at the module:class that holds your handler.")
        print(f"decider: handler {settings.api.handler} from {handler_module.__file__}", file=sys.stderr)
    return constructor(settings.config.get(), settings.api.pipeline, settings.api.mode,
                       sample_request=os.path.join(code_path, "sample_request.json"))
