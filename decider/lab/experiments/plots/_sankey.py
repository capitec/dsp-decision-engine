from __future__ import annotations

import json
import typing as t
from pathlib import Path

from decider.lab.experiments.metrics._paths import Flows
from decider.lab.experiments.plots._base import Plot

TEMPLATE = Path(__file__).with_name("sankey.vg.json")


class Sankey(Plot):
    """Record flow through branch arms into the outcome, from a `BranchFlows` metric, for one scenario.

    Vega-Lite has no Sankey, so this fills the Vega template `sankey.vg.json`:
    the input is the first stack, each branch arm sits in the stack of its
    depth, and the outcome values are the last stack.

    Example::

        Sankey(metric="branch_flows", scenario="limit_0.3")
    """

    type: t.Literal["sankey"] = "sankey"
    requires: t.ClassVar = Flows
    scenario: str = "baseline"

    def render(self, summaries: dict[str, Flows]) -> dict[str, t.Any]:
        flows = summaries.get(self.scenario)
        if flows is None:
            raise ValueError(f"no {self.scenario!r} summary to draw; it failed or isn't a scenario")
        stacks = _stacks(flows)
        # The input's count already shows under its label, so it is labelled by what it is.
        nodes = [{"category": n["id"], "label": "all records" if n["kind"] == "input" else n["label"],
                  "stack": stacks[n["id"]], "sort": i, "labels": "left" if n["kind"] == "input" else "right"}
                 for i, n in enumerate(flows.nodes)]
        links = [{"source": link["source"], "destination": link["target"], "value": link["value"]}
                 for link in flows.links]
        spec = json.loads(TEMPLATE.read_text())
        spec["title"]["text"] = f"{flows.outcome}: {self.scenario}"
        spec.update(self.size())
        spec["data"][0]["values"] = nodes + links
        return spec


def _stacks(flows: Flows) -> dict[str, int]:
    # Longest path from the input: a nested arm sits one stack past the arm it's inside.
    depth = {n["id"]: 1 for n in flows.nodes}
    for _ in flows.nodes:
        for link in flows.links:
            depth[link["target"]] = max(depth[link["target"]], depth[link["source"]] + 1)
    outcomes = [n["id"] for n in flows.nodes if n["kind"] == "outcome"]
    last = max([depth[i] for i in depth if i not in outcomes] + [1]) + 1
    return {**depth, **dict.fromkeys(outcomes, last)}
