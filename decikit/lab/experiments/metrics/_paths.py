from __future__ import annotations

import typing as t
from collections import Counter

import numpy as np
import polars as pl

from decikit.ir import BranchNode, iter_nodes, to_ir
from decikit.lab.experiments.callbacks import Batch, NodeBatch, Run
from decikit.lab.experiments.metrics._base import Metric, Summary

_ROW = "__row"


class Flows(Summary):
    """Sankey data: an input node, one node per branch arm and per outcome value, and record counts between them."""

    outcome: str
    nodes: list[dict[str, t.Any]]
    links: list[dict[str, t.Any]]


class BranchFlows(Metric):
    """How records flow through the flow's branch arms into each value of `outcome`.

    Example::

        BranchFlows(outcome="approved")
    """

    type: t.Literal["branch_flows"] = "branch_flows"
    summary: t.ClassVar = Flows
    outcome: str
    nodes: str | None = "*"

    def reset(self) -> None:
        self._visits: list[pl.DataFrame] = []
        self._links: Counter[tuple[str, str]] = Counter()
        self._values: dict[str, t.Any] = {}
        self._records = 0

    def on_run_start(self, run: Run) -> None:
        # Arm path -> (node id, label, start order); arms of one branch never share a record.
        self._arms = {
            arm.origin.path: (f"{node.origin.path}#{i}", arm.origin.path.split("/")[-1], order)
            for order, node in enumerate(iter_nodes(to_ir(run.step))) if isinstance(node, BranchNode)
            for i, arm in enumerate(node.arms)
        }

    def on_node_end(self, node: NodeBatch) -> None:
        arm = self._arms.get(node.path)
        if arm is not None and node.rows is not None:
            self._visits.append(pl.DataFrame({_ROW: node.rows, "iteration": node.iteration or 0,
                                              "order": arm[2], "arm": arm[0]}))

    def on_batch_end(self, batch: Batch) -> None:
        n = batch.output.height
        self._records += n
        ends = pl.DataFrame({_ROW: np.arange(n), "value": batch.output[self.outcome]})
        if self._visits:
            visits = pl.concat(self._visits).sort(_ROW, "iteration", "order").with_columns(
                source=pl.col("arm").shift(1).over(_ROW).fill_null("in"))
            for source, target, count in visits.group_by("source", "arm").len().iter_rows():
                self._links[(source, target)] += count
            ends = ends.join(visits.group_by(_ROW).agg(pl.col("arm").last()), on=_ROW, how="left")
        else:
            ends = ends.with_columns(arm=pl.lit(None, pl.String))
        for source, value, count in ends.group_by(pl.col("arm").fill_null("in"), "value").len().iter_rows():
            self._links[(source, f"out:{value}")] += count
            self._values[f"out:{value}"] = value
        self._visits = []

    def summarise(self) -> Flows:
        labels = {node_id: label for node_id, label, _ in self._arms.values()}
        seen = {end for link in self._links for end in link}
        nodes = [{"id": "in", "label": f"{self._records} records", "kind": "input"}]
        nodes += [{"id": i, "label": labels[i], "kind": "arm"} for i in sorted(seen & labels.keys())]
        nodes += [{"id": i, "label": f"{self.outcome} {self._values[i]}", "kind": "outcome"}
                  for i in sorted(self._values)]
        links = [{"source": s, "target": t, "value": v} for (s, t), v in sorted(self._links.items())]
        return Flows(outcome=self.outcome, nodes=nodes, links=links)
