from __future__ import annotations

import typing as t

from decider.experiments.metrics.compare import Divergences
from decider.experiments.metrics.outputs import Counts
from decider.experiments.plots.base import Plot, altair


class Bars(Plot):
    """Records per value of a `ValueCounts` metric, one bar per scenario.

    Example::

        Bars(metric="value_counts")
    """

    type: t.Literal["bars"] = "bars"
    requires: t.ClassVar = Counts

    def render(self, summaries: dict[str, Counts]):
        alt = altair()
        rows = [{"scenario": name, "value": value, "records": n}
                for name, s in summaries.items() for value, n in s.counts.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        return alt.Chart(alt.Data(values=rows), title=column).mark_bar().encode(
            x=alt.X("value:N", title=column),
            xOffset="scenario:N",
            y="records:Q",
            color="scenario:N",
            tooltip=["scenario:N", "value:N", "records:Q"],
        )


class Changes(Plot):
    """Records each scenario changed against the baseline, from a `Divergence` metric.

    Example::

        Changes(metric="divergence")
    """

    type: t.Literal["changes"] = "changes"
    requires: t.ClassVar = Divergences

    def render(self, summaries: dict[str, Divergences]):
        alt = altair()
        rows = [{"scenario": name, "changed": s.changed, "compared": s.compared, "first": s.first}
                for name, s in summaries.items()]
        column = next(iter(summaries.values())).column if summaries else ""
        return alt.Chart(alt.Data(values=rows), title=f"{column} changed vs baseline").mark_bar().encode(
            x=alt.X("scenario:N", sort=None),
            y=alt.Y("changed:Q", title="records changed"),
            tooltip=["scenario:N", "changed:Q", "compared:Q", alt.Tooltip("first:Q", title="first changed record")],
        )
