from __future__ import annotations

import typing as t

from decider.experiments.metrics.base import Summary
from decider.registry import BaseRegistryModule, ref


class Plot(BaseRegistryModule, root=True):
    """A pluggable plot of one metric's summaries, one per scenario.

    `metric` names the metric it draws (its `name`, or its type tag when
    unnamed); `requires` is the `Summary` type it can draw, checked before the
    run starts. `render` returns a JSON payload: a Vega-Lite spec, or data the
    UI draws.

    Example::

        class Totals(Plot):
            type: t.Literal["totals"] = "totals"
            requires: t.ClassVar = Total

            def render(self, summaries):
                return {scenario: s.n for scenario, s in summaries.items()}
    """

    requires: t.ClassVar[type[Summary]] = Summary
    metric: str
    name: str = ""

    @property
    def key(self) -> str:
        return self.name or self.type

    def render(self, summaries: dict[str, Summary]) -> dict[str, t.Any]:
        raise NotImplementedError


PlotRef = ref(Plot)
"""A pydantic field type holding any registered `Plot`, chosen by its `type` tag."""
