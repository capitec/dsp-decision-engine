from __future__ import annotations

import typing as t

from decikit.lab.experiments.metrics._compare import Divergences
from decikit.lab.experiments.metrics._outputs import Counts
from decikit.lab.experiments.plots._base import Plot, require_altair


class Bars(Plot):
    """Records per value of a `ValueCounts` metric, per scenario.

    `layout` places the scenarios: `"side_by_side"` next to each other for
    each value, `"stacked"` on top of each other, or `"rows"` one chart per
    scenario.

    Example::

        Bars(metric="value_counts", layout="stacked")
    """

    type: t.Literal["bars"] = "bars"
    requires: t.ClassVar = Counts
    layout: t.Literal["side_by_side", "stacked", "rows"] = "side_by_side"

    def render(self, summaries: dict[str, Counts]):
        alt = require_altair()
        rows = [{"scenario": name, "value": value, "records": n}
                for name, s in summaries.items() for value, n in s.counts.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        chart = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
            x=alt.X("value:N", title=column), y="records:Q", color="scenario:N",
            tooltip=["scenario:N", "value:N", "records:Q"])
        if self.layout == "side_by_side":
            chart = chart.encode(xOffset="scenario:N")
        if self.layout == "rows":
            return chart.properties(**{"height": 90, **self.size()}).facet(
                row=alt.Row("scenario:N", sort=list(summaries), title=None), title=column)
        return chart.properties(title=column, **self.size())


class Changes(Plot):
    """Records each scenario changed against the baseline, from a `Divergence` metric, one bar per scenario.

    Example::

        Changes(metric="divergence")
    """

    type: t.Literal["changes"] = "changes"
    requires: t.ClassVar = Divergences

    def render(self, summaries: dict[str, Divergences]):
        alt = require_altair()
        rows = [{"scenario": name, "changed": s.changed, "compared": s.compared, "first": s.first,
                 "share": s.changed / s.compared if s.compared else 0.0} for name, s in summaries.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        return alt.Chart(alt.Data(values=rows), title=f"{column} changed vs baseline").mark_bar().encode(
            y=alt.Y("scenario:N", sort=None, title=None),
            x=alt.X("changed:Q", title="records changed"),
            tooltip=["scenario:N", "changed:Q", "compared:Q", alt.Tooltip("share:Q", format=".1%"),
                     alt.Tooltip("first:Q", title="first changed record")],
        ).properties(**{"width": 360, **self.size()})
