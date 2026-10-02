from __future__ import annotations

import typing as t

from decider.experiments.metrics.compare import Divergences
from decider.experiments.metrics.outputs import Counts
from decider.experiments.plots.base import Plot

VEGA_LITE = "https://vega.github.io/schema/vega-lite/v5.json"


class Bars(Plot):
    """Records per value of a `ValueCounts` metric, one bar per scenario, as a Vega-Lite spec.

    Example::

        Bars(metric="value_counts")
    """

    type: t.Literal["bars"] = "bars"
    requires: t.ClassVar = Counts

    def render(self, summaries: dict[str, Counts]) -> dict[str, t.Any]:
        rows = [{"scenario": name, "value": value, "records": n}
                for name, s in summaries.items() for value, n in s.counts.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        return {
            "$schema": VEGA_LITE, "title": column, "data": {"values": rows}, "mark": "bar",
            "encoding": {
                "x": {"field": "value", "type": "nominal", "title": column},
                "xOffset": {"field": "scenario"},
                "y": {"field": "records", "type": "quantitative"},
                "color": {"field": "scenario", "type": "nominal"},
                "tooltip": [{"field": "scenario"}, {"field": "value"}, {"field": "records"}],
            },
        }


class Changes(Plot):
    """Records each scenario changed against the baseline, from a `Divergence` metric, as a Vega-Lite spec.

    Example::

        Changes(metric="divergence")
    """

    type: t.Literal["changes"] = "changes"
    requires: t.ClassVar = Divergences

    def render(self, summaries: dict[str, Divergences]) -> dict[str, t.Any]:
        rows = [{"scenario": name, "changed": s.changed, "compared": s.compared, "first": s.first}
                for name, s in summaries.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        return {
            "$schema": VEGA_LITE, "title": f"{column} changed vs baseline", "data": {"values": rows},
            "mark": "bar",
            "encoding": {
                "x": {"field": "scenario", "type": "nominal", "sort": None},
                "y": {"field": "changed", "type": "quantitative", "title": "records changed"},
                "tooltip": [{"field": "scenario"}, {"field": "changed"}, {"field": "compared"},
                            {"field": "first", "title": "first changed record"}],
            },
        }
