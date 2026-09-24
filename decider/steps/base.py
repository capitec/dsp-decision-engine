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
    from decider.engine.run.engine import Mode
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
        produced and consumed inside the step follow its writes. Relabels
        chain: a key may be the original name or the current one. A key the
        step doesn't read or write is an error.

        Example::

            ratio.relabel(reads={"instalment": "monthly_instalment"}, writes={"ratio": "dti"})
            ratio.relabel(writes={"ratio": "dti"}).relabel(writes={"dti": "dti_pct"})  # writes dti_pct
        """
        from decider.engine.ir.context import IRContext
        from decider.engine.ir.nodes import CallNode, iter_nodes
        from decider.engine.wiring.interface import interface

        node = IRContext().build(self)
        calls = [n for n in iter_nodes(node) if isinstance(n, CallNode)]
        # Unknown lineage: any name may be real, so nothing can be checked.
        known_reads = None if any(c.inputs is None for c in calls) else interface(node)[0]
        known_writes = None if any(c.outputs is None for c in calls) else {o.name for c in calls for o in c.outputs}
        return self._replace(
            reads=_compose(self, "reads", self.reads, reads, known_reads),
            writes=_compose(self, "writes", self.writes, writes, known_writes),
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
        """Run the step over a polars frame, interpreted; `kwargs` go to `Executable.run` (e.g. `params=`).

        For repeated runs, bind once with `Engine().bind(step)` instead.

        Example::

            out = pipeline.run(df, params={"term": {"cap_by_income": {"cap": 36.0}}})
        """
        from decider.engine import Engine

        return Engine().bind(self).run(data, **kwargs)

    def session(self, data: Any, params: Any = None, mode: Mode = "interpreted", **engine_kwargs: Any) -> Any:
        """Open a debug `Session` over a polars frame; `engine_kwargs` go to `Engine` (e.g. `params_validation=`).

        `mode` is the `Engine.bind` mode; `"fused"` pauses at kernel boundaries only.

        Example::

            s = pipeline.session(df, mode="stepped")
            s.break_at("term/cap_by_income")
            s.resume()
            s.set("term_cap", 36.0)
            s.resume()
            s.output()
        """
        from decider.engine import Engine

        return Engine(**engine_kwargs).bind(self, mode).session(data, params)


def _walk(step: Step, parent: str) -> Iterator[tuple[str, Step]]:
    path = parent
    if step.name is not None:
        path = f"{parent}/{step.name}" if parent else step.name
        yield path, step
    for member in getattr(step, "steps", ()):
        if isinstance(member, Step):
            yield from _walk(member, path)


def _compose(step: Step, kind: str, old: tuple, new: dict | None, known: set[str] | None) -> tuple:
    from decider.exceptions import WiringError
    from decider.registry.resolve import hint

    mapping = dict(old)
    current = {v: k for k, v in mapping.items()}
    for key, target in (new or {}).items():
        if key in current:
            mapping[current[key]] = target
        elif key in mapping or known is None or key in known:
            mapping[key] = target
        else:
            raise WiringError(
                f"{step.name or type(step).__name__}.relabel({kind}={{{key!r}: {target!r}}}): it {kind} no "
                f"{key!r}; it {kind} {sorted(known)}.{hint(key, known)}"
            )
    return tuple(mapping.items())


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
