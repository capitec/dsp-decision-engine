from __future__ import annotations

from typing import Iterator

from decider.engine.run.params import RunParams
from decider.engine.run.runners.base import Checkpoint
from decider.engine.run.runners.interpreted import _Scope
from decider.engine.run.runners.stepped import SteppedRunner
from decider.engine.run.state import State
from decider.engine.wiring.plan import Call, Sequence


class FusedRunner(SteppedRunner):
    """Runs each innermost sequence of scalar steps as one numba kernel; pauses at kernel boundaries only.

    A kernel covering several steps yields one `before`/`after` pair, with
    the origin of its first step; the steps after it in that kernel yield
    nothing. Row nodes are kernels of their own, and frame steps, branches
    and loops run in Python, as in stepped mode.

    Example::

        exe = Engine().bind(pipeline, mode="fused")
        exe.run(df)
    """

    fuse = True

    def _sequence(self, seq: Sequence, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        for child in seq.children:
            unit = self.units.get(child.id) if isinstance(child, Call) else None
            if unit is None or len(unit.calls) == 1:
                yield from self._node(child, state, params, scope)
            elif unit.calls[0] is child:
                origin = child.node.origin
                yield Checkpoint(origin, "before")
                self._run(unit, state, params, scope)
                yield Checkpoint(origin, "after")
