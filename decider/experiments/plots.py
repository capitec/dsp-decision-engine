"""Pluggable, registry-discoverable experiment plots.

A client defines their own plot by subclassing `BasePlotType` and
implementing `visualize`; naming it in `ExperimentDef.plots` (by alias or
import path, same as a `ConfigurableStep`) resolves it with no import needed
in the experiment asset itself. `Sankey` is today's built-in record-flow
graph, ported onto this registry unchanged; `OutcomeCounts` is a second,
simpler example showing how little a custom plot needs.

Every built-in plot here reads the baseline's already-captured trace (the
same `TraceSink` a run already pays for) rather than a live hook into the
runner: a hook fires from Python on every node a run reaches, which forces
the run off the compiled kernel entirely (see
`notes/experiment-plot-registry-spike.md` for the measured cost — about 700x
against the compiled path, regardless of how the hook is dispatched). That
tradeoff may be worth it for a one-off debugging plot; it isn't the default.
"""
from __future__ import annotations

import typing as t

from decider.experiments.graphs import path_sankey
from decider.registry import BaseRegistryModule, ref


class BasePlotType(BaseRegistryModule, root=True):
    """A named, pluggable plot: `visualize` turns a finished baseline run into an interactive payload.

    Example::

        class OutcomeCounts(BasePlotType):
            type: t.Literal["outcome_counts"] = "outcome_counts"
            def visualize(self, step, frame, params, baseline, outputs):
                ...
    """

    def visualize(self, step, frame, params, baseline: dict, outputs: tuple[str, ...]) -> dict[str, t.Any]:
        raise NotImplementedError


class Sankey(BasePlotType):
    """Record flow through the baseline's branches, ending at the outcome — see `graphs.path_sankey`."""

    type: t.Literal["sankey"] = "sankey"

    def visualize(self, step, frame, params, baseline: dict, outputs: tuple[str, ...]) -> dict[str, t.Any]:
        return path_sankey(step, frame, params, baseline, outputs)


class OutcomeCounts(BasePlotType):
    """How many baseline records landed on each value of one output column.

    Example::

        OutcomeCounts(type="outcome_counts", column="approved")
    """

    type: t.Literal["outcome_counts"] = "outcome_counts"
    column: str

    def visualize(self, step, frame, params, baseline: dict, outputs: tuple[str, ...]) -> dict[str, t.Any]:
        values = (baseline.get("output") or {}).get(self.column, [])
        counts: dict[str, int] = {}
        for v in values:
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        return {"kind": "bar", "column": self.column, "series": counts}


PlotRef = ref(BasePlotType)
"""A pydantic field type holding any registered `BasePlotType`, chosen by its `type` tag."""
