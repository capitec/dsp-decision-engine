from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Literal, Protocol

from decider.engine.ir.origin import Origin
from decider.engine.wiring.plan import Plan

if TYPE_CHECKING:
    from decider.engine.run.params import RunParams
    from decider.engine.run.state import State


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A point a runner stops at: just `before` a node runs, or just `after`.

    Every node a run reaches (calls, sequences, branches, loops) yields one
    `before` and one `after`, whatever the row count. Nodes inside an arm no
    row takes yield nothing; a loop body yields once per iteration.

    Args:
        arm: inside a branch arm, the arm's index (of the innermost branch).
        iteration: inside a loop, the iteration from 1 (of the innermost loop);
            the condition checked before iteration `k` carries `k` too.
    """

    origin: Origin
    when: Literal["before", "after"]
    arm: int | None = None
    iteration: int | None = None


class Runner(Protocol):
    """Executes a `Plan` over a `State`, one checkpoint at a time.

    Draining the generator runs the whole plan; a debug session advances it
    one checkpoint at a time and may inspect or change `state` in between.

    Example::

        for checkpoint in InterpretedRunner().iterate(plan, state, params):
            print(checkpoint.when, checkpoint.origin.path)
    """

    def iterate(self, plan: Plan, state: State, params: RunParams) -> Iterator[Checkpoint]: ...
