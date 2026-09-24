from __future__ import annotations

from typing import Iterable, Iterator

from decider.engine.compile import FALLBACK_ERRORS, ArmOutOfRange, Packed, compile_packed
from decider.engine.run.params import RunParams
from decider.engine.run.runners.base import Checkpoint
from decider.engine.run.runners.interpreted import _Scope
from decider.engine.run.runners.stepped import SteppedRunner, _external
from decider.engine.run.state import State
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Sequence


class FusedRunner(SteppedRunner):
    """Runs each innermost sequence of scalar steps as one numba kernel; pauses at kernel boundaries only.

    A kernel covering several steps yields one `before`/`after` pair, with
    the origin of its first step; the steps after it in that kernel yield
    nothing. A branch or loop of plain scalar steps (nested ones included)
    is one kernel too, with the branch's or loop's checkpoints, so nothing
    inside it yields; `packed` maps its path to the kernel. Row nodes are
    kernels of their own, and frame steps, other branches and loops run in
    Python, as in stepped mode.

    Example::

        exe = Engine().bind(pipeline, mode="fused")
        exe.run(df)
    """

    fuse = True

    def __init__(self) -> None:
        super().__init__()
        self.packed: dict[str, Packed] = {}

    def _compile(self, plan: Plan, lazy: bool) -> None:
        super()._compile(plan, lazy)
        self.packed = compile_packed(plan, lazy, self._python)
        self._reads.update((id(k), _external(k)) for k in self.packed.values())

    def _sequence(self, seq: Sequence, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        for child in seq.children:
            unit = self.units.get(child.id) if isinstance(child, Call) else None
            if unit is None or len(unit.calls) == 1:
                yield from self._node(child, state, params, scope)
            elif unit.calls[0] is child:
                if self.skip and (passed := self.skip.get(child)) is not None:
                    scope.names.update((v.name, v) for v in passed)
                    continue
                origin = child.node.origin
                yield scope.checkpoint(origin, "before")
                self._run(unit, state, params, scope)
                yield scope.checkpoint(origin, "after")

    def _branch(self, branch: Branch, state: State, params: RunParams, scope: _Scope) -> Iterable[Checkpoint]:
        return () if self._packed(branch, state, params, scope) else super()._branch(branch, state, params, scope)

    def _loop(self, loop: Loop, state: State, params: RunParams, scope: _Scope) -> Iterable[Checkpoint]:
        return () if self._packed(loop, state, params, scope) else super()._loop(loop, state, params, scope)

    def _packed(self, r: Branch | Loop, state: State, params: RunParams, scope: _Scope) -> bool:
        path = r.node.origin.path
        kernel = self.packed.get(path)
        if kernel is None:
            return False
        for v in kernel.guarded:
            mask = state.valid.get(v.id)
            absent = v.id not in state.values
            if absent or mask is not None and not (mask if scope.rows is None else mask[scope.rows]).all():
                return False
        try:
            self._run(kernel, state, params, scope)
        except ArmOutOfRange:
            # The unpacked path raises it again, naming the branch, the arm and the rows.
            return False
        except FALLBACK_ERRORS:
            # ponytail: dropped for the rest of this runner's life; retry per call if a kernel only fails on some inputs.
            self.packed.pop(path, None)
            return False
        return True
