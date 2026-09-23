from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from decider.engine.ir.decls import Input
from decider.engine.ir.nodes import BranchNode, CallNode, LoopNode, SequenceNode


@dataclass(frozen=True, slots=True, eq=False)
class Version:
    """One written value of a name: the state a runner stores and a reader reads.

    Args:
        id: position in `Plan.versions`; stable for one IR, never derived from paths.
        producer: origin path of the node that wrote it; `None` for a pipeline
            input column. A branch's merged value is produced by the branch, a
            loop's carried value by the loop, a name read after an
            unknown-lineage frame step by that frame step.

    Example::

        Version(3, "term_cap", "term/cap_by_income", float)
    """

    id: int
    name: str
    producer: str | None
    annotation: Any = None


@dataclass(slots=True, eq=False)
class Call:
    """A `CallNode` with its id and the versions it reads and writes.

    `reads[i]` feeds `node.inputs[i]` and `writes[i]` receives `node.outputs[i]`.
    `reads` is `None` when the node reads the whole frame (a frame step of
    unknown inputs). For a frame step of unknown outputs (a barrier), `writes`
    lists the names later nodes read from it; the runner checks each is a
    column of the frame it returns.
    """

    id: int
    node: CallNode
    reads: tuple[Version, ...] | None
    writes: tuple[Version, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Sequence:
    """A `SequenceNode` whose children are resolved, in execution order."""

    node: SequenceNode
    children: tuple[Resolved, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Merge:
    """How a branch combines one `modifies` name.

    On each row, `version` takes the value of the taken arm's `arms[k]`, or
    `prior` when that arm leaves the name alone (`arms[k] is None`). `prior`
    is `None` only when every arm writes the name.
    """

    version: Version
    prior: Version | None
    arms: tuple[Version | None, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Branch:
    """A `BranchNode` resolved: its condition, its arms and one `Merge` per `modifies` name."""

    node: BranchNode
    condition: Call
    arms: tuple[Resolved, ...]
    merges: tuple[Merge, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Carry:
    """One `carries` name of a loop.

    `version` starts as `initial`, is what the condition and body read in
    each iteration, is set from `last` at the end of each iteration, and is
    what nodes after the loop read.
    """

    version: Version
    initial: Version
    last: Version


@dataclass(frozen=True, slots=True, eq=False)
class Loop:
    """A `LoopNode` resolved: its condition, body and carries."""

    node: LoopNode
    condition: Call
    body: Resolved
    carries: tuple[Carry, ...]


Resolved = Union[Call, Sequence, Branch, Loop]


@dataclass(frozen=True, slots=True, eq=False)
class Plan:
    """An IR with every name bound to a version: what runners and the compiler execute.

    Args:
        root: the resolved tree, same shape as the IR.
        calls: every `Call` in execution order (a condition before its arms);
            `calls[i].id == i`.
        versions: every `Version`; `versions[i].id == i`.
        inputs: the pipeline input columns, as declared by their first reader.
            Their versions are the ones with `producer is None`.
        outputs: output column name -> version: input columns, values nothing
            reads, and emitted values (`"term_cap@term/cap_by_income"` for a
            qualified emit). Frame columns no node reads pass through too,
            unless dropped.
        drops: names to leave out of the output, frame columns included.
        chains: every produced version of each name, in production order.

    Example::

        plan = resolve(pipeline)
        [v.producer for v in plan.chains["term_cap"]]
    """

    root: Resolved
    calls: tuple[Call, ...]
    versions: tuple[Version, ...]
    inputs: tuple[Input, ...]
    outputs: dict[str, Version]
    drops: tuple[str, ...]
    chains: dict[str, tuple[Version, ...]]
