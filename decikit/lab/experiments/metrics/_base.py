from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict

from decikit.lab.experiments.callbacks import Callback
from decikit._registry import BaseRegistryModule, ref


class Summary(BaseModel):
    """One metric's result for one scenario. Subclass it per kind of result, so a plot can require one."""

    model_config = ConfigDict(extra="allow")


class Metric(BaseRegistryModule, Callback, root=True, group="metrics"):
    """A pluggable collector: its hooks accumulate one scenario's batches, `summarise` reduces them.

    The runner gives every scenario its own copy and calls `reset` on it
    first. `name` keys its summaries in the result and is what a plot names
    (the type tag when empty). `nodes`, a glob on node paths, or `node_id`, a
    step's durable id, makes it a step metric: `on_node_end` fires for the
    matching steps. Without either it is global and reads the flow's output.
    `summary` is the `Summary` type `summarise` returns.

    Example::

        class RowCount(Metric):
            type: t.Literal["row_count"] = "row_count"
            summary: t.ClassVar = Total

            def reset(self):
                self._n = 0

            def on_batch_end(self, batch):
                self._n += batch.output.height

            def summarise(self):
                return Total(n=self._n)
    """

    summary: t.ClassVar[type[Summary]] = Summary
    name: str = ""
    nodes: str | None = None
    node_id: str | None = None

    @property
    def key(self) -> str:
        return self.name or self.type

    @property
    def on_steps(self) -> bool:
        """Whether it reads steps (`nodes` or `node_id`) rather than the flow's output."""
        return self.nodes is not None or self.node_id is not None

    def reset(self) -> None:
        pass

    def summarise(self) -> Summary:
        raise NotImplementedError


MetricRef = ref(Metric)
"""A pydantic field type holding any registered `Metric`, chosen by its `type` tag."""
