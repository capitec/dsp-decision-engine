from __future__ import annotations

import typing as t

from decider.experiments.metrics.outputs import Bins, Stats
from decider.experiments.plots.base import Plot, altair


class Distribution(Plot):
    """A `Histogram` metric's bins, one translucent histogram per scenario; click the legend to pick one.

    Records below or above the bins show as one more bin at either end.

    Example::

        Distribution(metric="histogram")
    """

    type: t.Literal["distribution"] = "distribution"
    requires: t.ClassVar = Bins

    def render(self, summaries: dict[str, Bins]):
        alt = altair()
        rows = [row for name, s in summaries.items() for row in _bins(name, s)]
        column = next(iter(summaries.values())).column if summaries else ""
        pick = alt.selection_point(fields=["scenario"], bind="legend")
        return alt.Chart(alt.Data(values=rows), title=f"{column} by scenario").mark_bar().encode(
            x=alt.X("start:Q", bin="binned", title=column),
            x2="end:Q",
            y=alt.Y("records:Q", stack=None),
            color="scenario:N",
            opacity=alt.condition(pick, alt.value(0.55), alt.value(0.05)),
            tooltip=["scenario:N", alt.Tooltip("start:Q", format=".4g"), alt.Tooltip("end:Q", format=".4g"),
                     "records:Q"],
        ).add_params(pick)


class Ranges(Plot):
    """A `Describe` metric per scenario: a line from min to max with a point at the mean.

    Example::

        Ranges(metric="describe")
    """

    type: t.Literal["ranges"] = "ranges"
    requires: t.ClassVar = Stats

    def render(self, summaries: dict[str, Stats]):
        alt = altair()
        rows = [{"scenario": name, "min": s.min, "mean": s.mean, "max": s.max, "count": s.count}
                for name, s in summaries.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        base = alt.Chart(alt.Data(values=rows), title=f"{column}: min, mean and max").encode(
            y=alt.Y("scenario:N", sort=None))
        return base.mark_rule().encode(x=alt.X("min:Q", title=column), x2="max:Q") + base.mark_point(
            filled=True, size=80).encode(x="mean:Q", tooltip=["scenario:N", "min:Q", "mean:Q", "max:Q", "count:Q"])


def _bins(scenario: str, s: Bins) -> list[dict[str, t.Any]]:
    rows = [{"scenario": scenario, "start": s.edges[i], "end": s.edges[i + 1], "records": n}
            for i, n in enumerate(s.counts)]
    if rows and s.below:
        width = s.edges[1] - s.edges[0]
        rows.insert(0, {"scenario": scenario, "start": s.edges[0] - width, "end": s.edges[0], "records": s.below})
    if rows and s.above:
        width = s.edges[-1] - s.edges[-2]
        rows.append({"scenario": scenario, "start": s.edges[-1], "end": s.edges[-1] + width, "records": s.above})
    return rows
