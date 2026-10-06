"""Work on a flow away from serving: load data, check it, compare versions, test it, run experiments.

- `data`: read datasets and the project's dataset catalog.
- `checks`: static checks over a flow and the findings report (`decider check`).
- `diff`: compare two versions by structure, outputs over data, git revisions or proof (`decider diff`).
- `testing`: assertions for your own test suite: `assert_equivalent`, `no_recompile`, `corpus`.
- `experiments`: run a baseline and scenarios over the same records and measure them (`decider experiment`).

The names here are how all of them refer to a flow's parts: `describe(pipeline)` gives a
`FlowDescription` of `StepRef` nodes, `EdgeRef` edges and `ValueSlotRef` values, and a check
finding or an experiment result points back at a step by its `StepRef`.

Example::

    from decider.lab import checks, data, describe
    frame = data.read("loans.parquet")
    checks.run(pipeline).ok
    [n.path for n in describe(pipeline).nodes]
"""
from decider.lab._describe import describe, resolve_step, source_location
from decider.lab._refs import CONTRACT_VERSION, EdgeRef, FlowDescription, FlowRef, RecordRef, StepRef, ValueSlotRef

__all__ = [
    "CONTRACT_VERSION", "EdgeRef", "FlowDescription", "FlowRef", "RecordRef", "StepRef", "ValueSlotRef", "describe",
    "resolve_step", "source_location",
]
