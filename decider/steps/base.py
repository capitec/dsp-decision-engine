from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

from decider.engine.ir.origin import check_name

if TYPE_CHECKING:
    from typing import Self

    from decider.engine.ir.context import IRContext
    from decider.engine.ir.nodes import IRNode
    from decider.engine.params.schema import ParamsSchema
    from decider.steps.sequential import SequentialStep


class Step(ABC):
    """Base of everything that can sit in a pipeline.

    Steps combine with `|` (in written order), `flow`, `dag`, `branch` and
    `loop`; a plain function converts automatically wherever a step is
    expected. Every step turns itself into IR through `to_ir`.

    Example::

        pipeline = step(disposable_income) | ratio | affordable
        pipeline.parameters().defaults()
    """

    __slots__ = ("__weakref__",)

    def __post_init__(self) -> None:
        if self.name is not None:
            check_name(self.name)

    @abstractmethod
    def to_ir(self, ctx: IRContext) -> IRNode:
        """This step's IR, placed under `ctx`. Name nodes with `ctx.origin(self)`."""

    def _replace(self, **changes: Any) -> Self:
        return dataclasses.replace(self, **changes)

    def __or__(self, other: Any) -> SequentialStep:
        from decider.steps.sequential import flow

        return flow(self, other)

    def __ror__(self, other: Any) -> SequentialStep:
        from decider.steps.sequential import flow

        return flow(other, self)

    def named(self, name: str) -> Self:
        """A copy of this step under another name, e.g. to place it twice.

        Example::

            flow(cap, cap.named("cap_again"))
        """
        return self._replace(name=check_name(name))

    def relabel(self, *, reads: dict[str, str] | None = None, writes: dict[str, str] | None = None) -> Self:
        """A copy that reads and writes other names at its boundary.

        `reads` maps a name the step reads to the column it should read
        instead; `writes` maps a name it writes to the name to write. Names
        produced and consumed inside the step follow its writes.

        Example::

            ratio.relabel(reads={"instalment": "monthly_instalment"}, writes={"ratio": "dti"})
        """
        return self._replace(
            reads=tuple({**dict(self.reads), **(reads or {})}.items()),
            writes=tuple({**dict(self.writes), **(writes or {})}.items()),
        )

    def parameters(self) -> ParamsSchema:
        """Every param this step needs, by node path, without running anything.

        Example::

            pipeline.parameters().defaults()   # a complete params document of defaults
        """
        from decider.engine import to_ir
        from decider.engine.params.schema import parameters

        return parameters(to_ir(self))

    def walk(self) -> Iterator[tuple[str, Step]]:
        """The authoring tree as `(path, step)` pairs; anonymous flows are transparent.

        Example::

            for path, s in pipeline.walk():
                print(path, type(s).__name__)
        """
        return _walk(self, "")

    def run(self, data: Any, **kwargs: Any) -> Any:
        """Run the step over a frame."""
        raise NotImplementedError("running steps needs the engine runner, which is not built yet")

    def session(self, data: Any, **kwargs: Any) -> Any:
        """Open a debug session over a frame."""
        raise NotImplementedError("debug sessions need the engine runner, which is not built yet")


def _walk(step: Step, parent: str) -> Iterator[tuple[str, Step]]:
    path = parent
    if step.name is not None:
        path = f"{parent}/{step.name}" if parent else step.name
        yield path, step
    for member in getattr(step, "steps", ()):
        if isinstance(member, Step):
            yield from _walk(member, path)


def as_step(value: Any) -> Step:
    """`value` as a step: steps pass through, functions become `step(fn)`.

    Example::

        as_step(ratio)  # FunctionStep(name="ratio", ...)
    """
    if isinstance(value, Step):
        return value
    if callable(value):
        from decider.steps.function import step

        return step(value)
    raise TypeError(f"{value!r} is not a step or a function")
