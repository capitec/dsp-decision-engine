from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

from pydantic import ConfigDict

from decikit._engine.run.engine import Executable
from decikit._engine.run.hidden import warn_hidden_state
from decikit._engine.run.runners import FusedRunner, InterpretedRunner, Runner, SteppedRunner
from decikit._engine.wiring import resolve
from decikit._registry import BaseRegistryModule


class Engine(BaseRegistryModule, root=True, group="engines"):
    """Turns a step (or its IR) into an `Executable`: a settings document, named by its `type` tag.

    The built-in tags are `"interpreted"`, `"stepped"` and `"fused"`; an
    installed package adds an engine as an entry point in `decikit.engines`.
    `Engine.resolve(tag)` gives the class, and `pipeline.compile(tag)` or
    `pipeline.compile(instance)` binds one.

    Args:
        params_validation: `"eager"` validates every node's params when a
            params document is first seen and raises on any invalid one;
            `"lazy"` validates a node the first time it runs, so an invalid
            param in a node no row reaches never fails a run.

    Example::

        exe = Engine.resolve("fused")(params_validation="lazy").bind(pipeline)
        out = exe.run(df, params={"term": {"cap_by_income": {"cap": 36.0}}})
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    params_validation: Literal["eager", "lazy"] = "eager"

    @abstractmethod
    def bind(self, step_or_ir: Any) -> Executable:
        """An `Executable` running `step_or_ir`: a step, a plain function or an IR node.

        Example::

            exe = FusedEngine().bind(pipeline)
        """


class NumbaEngine(Engine):
    """An engine that runs the plan through the numba driver; a subclass supplies the runner.

    Args:
        strict_compile: on the stepped and fused engines, a step that would
            run in Python raises instead of running, a step of a kernel numba
            refuses at launch included; mark the ones you accept
            `@allow_fallback`. On every engine, a step reading hidden state
            raises at `bind`.

    Example::

        class MyEngine(NumbaEngine):
            type: Literal["mine"] = "mine"

            def _runner(self):
                return MyRunner(strict=self.strict_compile)
    """

    strict_compile: bool = False

    @abstractmethod
    def _runner(self) -> Runner: ...

    def bind(self, step_or_ir: Any) -> Executable:
        exe = Executable(resolve(step_or_ir), self._runner(), self.params_validation == "lazy", step_or_ir,
                         engine=self)
        warn_hidden_state(exe.plan, strict=self.strict_compile)
        return exe


class InterpretedEngine(NumbaEngine):
    """Plain Python, step by step; a session pauses at every node and inside trees.

    Example::

        pipeline.compile("interpreted").run(df)
    """

    type: Literal["interpreted"] = "interpreted"

    def _runner(self) -> Runner:
        return InterpretedRunner()


class SteppedEngine(NumbaEngine):
    """One numba kernel per step; a session pauses at every node.

    Example::

        pipeline.session(df, engine="stepped")
    """

    type: Literal["stepped"] = "stepped"

    def _runner(self) -> Runner:
        return SteppedRunner(strict=self.strict_compile)


class FusedEngine(NumbaEngine):
    """One numba kernel per run of scalar steps: the production engine, and what serving runs by default.

    Example::

        pipeline.compile(FusedEngine(strict_compile=True)).run({"income": 1000.0, "debt": 200.0})
    """

    type: Literal["fused"] = "fused"

    def _runner(self) -> Runner:
        return FusedRunner(strict=self.strict_compile)


_FUSED = FusedEngine()
_default: ContextVar[Engine | None] = ContextVar("decikit_default_engine", default=None)


def default_engine() -> Engine:
    return _default.get() or _FUSED


def as_engine(which: str | Engine | None) -> Engine:
    if which is None:
        return default_engine()
    return Engine.resolve(which)() if isinstance(which, str) else which


@contextmanager
def engine(which: str | Engine) -> Iterator[None]:
    """Run every `pipeline.run(...)` and `pipeline.compile()` inside the block on `which`.

    `which` is an engine's tag (`"interpreted"`, `"stepped"`, `"fused"`) or an
    `Engine` instance. Outside any block the default is `FusedEngine()`. The
    setting is a context variable: it ends with the block (also on an
    exception), an `asyncio` task started inside gets a copy, and a thread
    sees it only when run under `contextvars.copy_context().run`.
    `pipeline.session(...)` opens interpreted whatever is set, and serving
    runs its own engine.

    Example::

        with decikit.engine("interpreted"):     # instant runs while editing
            pipeline.run(df)
        pipeline.run(df)                        # fused again
    """
    token = _default.set(as_engine(which))
    try:
        yield
    finally:
        _default.reset(token)
