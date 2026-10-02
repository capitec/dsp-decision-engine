from __future__ import annotations

import typing as t

from decider.experiments.metrics.paths import Flows
from decider.experiments.plots.base import Plot


class Sankey(Plot):
    """Record flow through branch arms into the outcome, from a `BranchFlows` metric, per scenario.

    Returns `{"kind": "sankey", "scenarios": {name: {"outcome", "nodes", "links"}}}` for the UI to draw.

    Example::

        Sankey(metric="branch_flows")
    """

    type: t.Literal["sankey"] = "sankey"
    requires: t.ClassVar = Flows

    def render(self, summaries: dict[str, Flows]) -> dict[str, t.Any]:
        return {"kind": "sankey", "scenarios": {name: s.model_dump() for name, s in summaries.items()}}
