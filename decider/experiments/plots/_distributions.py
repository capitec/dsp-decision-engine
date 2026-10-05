from __future__ import annotations

import typing as t

from decider.experiments.metrics._outputs import Bins, Stats
from decider.experiments.plots._base import Plot, require_altair

Layout = t.Literal["overlay", "stacked", "side_by_side", "rows"]


class Distribution(Plot):
    """A `Histogram` metric's bins per scenario; click the legend to pick a scenario out.

    `layout` places the scenarios: `"overlay"` translucent on one axis,
    `"stacked"` on top of each other, `"side_by_side"` next to each other in
    each bin, or `"rows"` one histogram per scenario. Records below or above
    the bins show as one more bin at either end.

    Example::

        Distribution(metric="histogram", layout="rows", width=500)
    """

    type: t.Literal["distribution"] = "distribution"
    requires: t.ClassVar = Bins
    layout: Layout = "overlay"

    def render(self, summaries: dict[str, Bins]):
        alt = require_altair()
        rows = [row for name, s in summaries.items() for row in _bins(name, s)]
        column = next(iter(summaries.values())).column if summaries else ""
        pick = alt.selection_point(fields=["scenario"], bind="legend")
        tooltip = ["scenario:N", alt.Tooltip("start:Q", format=".4g"), alt.Tooltip("end:Q", format=".4g"), "records:Q"]
        base = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
            color="scenario:N", tooltip=tooltip,
            opacity=alt.condition(pick, alt.value(0.55 if self.layout == "overlay" else 0.9), alt.value(0.08)),
        ).add_params(pick)
        if self.layout == "side_by_side":
            order = list(dict.fromkeys(r["bin"] for r in sorted(rows, key=lambda r: r["start"])))
            chart = base.encode(x=alt.X("bin:N", sort=order, title=column), xOffset="scenario:N", y="records:Q")
        else:
            chart = base.encode(x=alt.X("start:Q", bin="binned", title=column), x2="end:Q",
                                y=alt.Y("records:Q", stack="zero" if self.layout == "stacked" else None))
        if self.layout == "rows":
            return chart.properties(**{"height": 90, **self.size()}).facet(
                row=alt.Row("scenario:N", sort=list(summaries), title=None), title=f"{column} by scenario")
        return chart.properties(title=f"{column} by scenario", **self.size())


class Ranges(Plot):
    """A `Describe` metric per scenario: a line from min to max with a point at the mean.

    Example::

        Ranges(metric="describe")
    """

    type: t.Literal["ranges"] = "ranges"
    requires: t.ClassVar = Stats

    def render(self, summaries: dict[str, Stats]):
        alt = require_altair()
        rows = [{"scenario": name, "min": s.min, "mean": s.mean, "max": s.max, "count": s.count}
                for name, s in summaries.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        base = alt.Chart(alt.Data(values=rows), title=f"{column}: min, mean and max").encode(
            y=alt.Y("scenario:N", sort=None))
        return (base.mark_rule().encode(x=alt.X("min:Q", title=column), x2="max:Q") + base.mark_point(
            filled=True, size=80).encode(x="mean:Q", tooltip=["scenario:N", "min:Q", "mean:Q", "max:Q", "count:Q"])
                ).properties(**self.size())


def _bins(scenario: str, s: Bins) -> list[dict[str, t.Any]]:
    edges = list(s.edges)
    counts = list(s.counts)
    if counts and s.below:
        edges.insert(0, edges[0] - (edges[1] - edges[0]))
        counts.insert(0, s.below)
    if counts and s.above:
        edges.append(edges[-1] + (edges[-1] - edges[-2]))
        counts.append(s.above)
    return [{"scenario": scenario, "start": edges[i], "end": edges[i + 1], "records": n,
             "bin": f"{edges[i]:.3g}–{edges[i + 1]:.3g}"} for i, n in enumerate(counts)]
