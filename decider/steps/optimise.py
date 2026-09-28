"""`optimise`: generate candidates, score each, keep the best per row.

A row-preserving step for the "try every candidate, keep the winner" shape:
best bundle, best term, best price point, best allocation. `optimise` lowers
to a plain `loop`, so it fuses into one kernel like any other loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from decider.engine.ir.nodes import SequenceNode
from decider.exceptions import WiringError
from decider.steps.base import Step, as_step
from decider.steps.function import step
from decider.steps.loop import loop
from decider.steps.sequential import flow
from decider.types import Struct, item_schema

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext

_CARRIES = ("index", "best_index", "best_score", "evaluated", "disqualified")

_ZERO = {float: 0.0, int: 0, bool: False}


def _more(index: int, count: int) -> bool:
    return index <= count


@step(outputs=_CARRIES, name="seed")
def _seed(count: int) -> tuple[int, int, float, int, int]:
    return 1, -1, -1e300, 0, 0


@step(output="index", name="advance")
def _advance(index: int) -> int:
    return index + 1


@step(output="reject", name="reject")
def _no_reject() -> bool:
    return False


@step(outputs=("best_index", "best_score", "evaluated", "disqualified"), name="keep")
def _keep(index: int, best_index: int, best_score: float, evaluated: int, disqualified: int,
          score: float, reject: bool) -> tuple[int, float, int, int]:
    if reject:
        disqualified += 1
    else:
        evaluated += 1
        if score > best_score or best_index == -1:
            best_index, best_score = index, score
    return best_index, best_score, evaluated, disqualified


def _write_as(step: Step, name: str) -> Step:
    from decider.engine import to_ir
    from decider.engine.wiring.interface import interface

    writes = interface(to_ir(step))[1]
    if len(writes) != 1:
        raise WiringError(f"optimise: expected one value, got {sorted(writes)}")
    own = next(iter(writes))
    return step if own == name else step.relabel(writes={own: name})


@dataclass(frozen=True, slots=True, eq=False)
class OptimiseStep(Step):
    """Runs `evaluate` once per candidate `index` in `1..count` and keeps the best. Build with `optimise()`."""

    __module__ = "decider.steps"

    name: str
    count: Step
    evaluate: Step
    score: str
    disqualify: Step | None
    max_candidates: int
    record: Any = None

    def to_ir(self, ctx: IRContext) -> SequenceNode:
        inner = ctx.child(self.name)
        keep = _keep if self.score == "score" else _keep.relabel(reads={"score": self.score})
        dq = _no_reject if self.disqualify is None else _write_as(self.disqualify, "reject")
        if self.record is None:
            body = flow(self.evaluate, dq, keep, _advance, name="body")
            search = loop(_more, body, carries=_CARRIES, max_iterations=self.max_candidates, name="search")
            return SequenceNode(ctx.origin(self), (inner.build(_write_as(self.count, "count")),
                                                   inner.build(_seed), inner.build(search)), drops=("index",))
        return self._with_record(inner, dq, keep, ctx)

    def _with_record(self, inner: Any, dq: Step, keep: Step, ctx: IRContext) -> SequenceNode:
        item = self.record
        schema = item_schema(item)
        zeros = tuple(_ZERO[t] for _, t in schema)

        def seed_record() -> Struct[item]:
            return zeros

        seed_record.__annotations__ = {"return": Struct[item]}

        def keep_record(best_record: Struct[item], score: float, best_score: float, reject: bool,
                        record: Struct[item]) -> Struct[item]:
            return record if not reject and score > best_score else best_record

        keep_record.__annotations__ = {"best_record": Struct[item], "score": float, "best_score": float,
                                       "reject": bool, "record": Struct[item], "return": Struct[item]}

        def finalise(best_index: int, best_record: Struct[item]) -> Struct[item] | None:
            return best_record if best_index >= 0 else None

        finalise.__annotations__ = {"best_index": int, "best_record": Struct[item], "return": Struct[item] | None}

        def pass_index(best_index: int) -> int:
            return best_index

        body = flow(self.evaluate, dq, step(keep_record, output="best_record", name="keep_record"),
                    keep, _advance, name="body")
        search = loop(_more, body, carries=(*_CARRIES, "best_record"),
                      max_iterations=self.max_candidates, name="search")
        return SequenceNode(ctx.origin(self), (inner.build(_write_as(self.count, "count")),
                                               inner.build(_seed),
                                               inner.build(step(seed_record, output="best_record", name="seed_record")),
                                               inner.build(search),
                                               inner.build(flow(step(finalise, output="record", name="finalise"),
                                                               step(pass_index, output="best_index", name="pass_index"),
                                                               name="finalise"))),
                            drops=("index",))


def optimise(count: Any, evaluate: Any, *, score: str = "score", disqualify: Any = None,
             max_candidates: int, name: str, record: Any = None) -> OptimiseStep:
    """Run `evaluate` once per candidate and keep the best one, per row.

    `evaluate` is a flow of ordinary steps that read `index` (the candidate,
    from 1 to `count`) plus the request data, and write `score` (and any other
    values). `count` is a step producing one int, the number of candidates
    (e.g. `(1 << len(loans)) - 1`). The candidate with the highest `score`
    wins; negate the score to minimise. A tie keeps the earlier candidate.

    `disqualify` is an optional bool step run after `evaluate`; a disqualified
    candidate is counted and excluded. To skip expensive work, have `evaluate`
    return `-inf` for a candidate you reject cheaply and disqualify `-inf`.

    The winner is `best_index` (`-1` when no candidate survived), its score
    `best_score` (`-1e300` when there is no winner), and the counts `evaluated`
    / `disqualified`.

    Args:
        score: the `evaluate` output to maximise (default `"score"`).
        max_candidates: the most candidates any row may test, the loop's bound.
            `count` per row may be less; a larger `count` than `max_candidates`
            simply stops there.
        record: an `Item` TypedDict naming the winner's record. With it,
            `evaluate` must also write a `record` output of `Struct[Item]`, and
            `optimise` emits that record for the winning candidate (`None` when
            no candidate survived).

    Example::

        @step(output="count")
        def bundles(items: Columnar[Item]) -> int:
            return (1 << len(items.price)) - 1

        @step(output="score")
        def margin(index: int, items: Columnar[Item]) -> float:
            total = 0.0
            for j in range(len(items.price)):
                if (index >> j) & 1:
                    total += items.price[j]
            return total

        best = optimise(bundles, flow(margin, name="evaluate"), max_candidates=1 << 12, name="best")
        out = flow(best, name="order").run(orders)   # best_index, best_score, evaluated, disqualified
    """
    if type(max_candidates) is not int or max_candidates < 1:
        raise WiringError(f"optimise {name!r}: max_candidates must be a positive int, got {max_candidates!r}")
    evaluate = as_step(evaluate)
    count = as_step(count)
    dq = None if disqualify is None else as_step(disqualify)
    from decider.engine import to_ir
    from decider.engine.wiring.interface import interface

    if score not in interface(to_ir(evaluate))[1]:
        raise WiringError(f"optimise {name!r}: evaluate does not write {score!r}; it writes "
                          f"{sorted(interface(to_ir(evaluate))[1])}")
    if record is not None:
        from decider.engine.compile.structs import struct_dtype, struct_schema
        struct_dtype(struct_schema(record))  # raises on a non-TypedDict or a non-float/int/bool field
        if "record" not in interface(to_ir(evaluate))[1]:
            raise WiringError(f"optimise {name!r}: evaluate does not write 'record'; it writes "
                              f"{sorted(interface(to_ir(evaluate))[1])}")
    return OptimiseStep(name, count, evaluate, score, dq, max_candidates, record)
