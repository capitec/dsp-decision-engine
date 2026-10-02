from __future__ import annotations

import typing as t

from decider.experiments import Bars, Changes, Divergence, Experiment, Plot, Stats, ValueCounts, Describe

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


def test_bars_draws_value_counts_per_scenario_as_vega_lite(pipeline, frame):
    result = (Experiment("x", pipeline, frame)
              .scenario("limit_0.3", params=LIMIT_03)
              .metric(ValueCounts(column="approved", nodes="demo/approved"))
              .plot(Bars(metric="value_counts"))
              .run())

    spec = result.plots["bars"]
    assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert {(r["scenario"], r["value"], r["records"]) for r in spec["data"]["values"]} == {
        ("baseline", "True", 3), ("baseline", "False", 1), ("limit_0.3", "True", 2), ("limit_0.3", "False", 2)}


def test_changes_draws_records_changed_per_scenario(pipeline, frame):
    result = (Experiment("x", pipeline, frame)
              .scenario("limit_0.3", params=LIMIT_03)
              .metric(Divergence(column="tier"))
              .plot(Changes(metric="divergence"))
              .run())
    assert [(r["scenario"], r["changed"]) for r in result.plots["changes"]["data"]["values"]] == [
        ("baseline", 0), ("limit_0.3", 1)]


def test_a_custom_plot_renders_the_summaries_of_the_metric_it_names(pipeline, frame):
    class Means(Plot):
        type: t.Literal["test_means"] = "test_means"
        requires: t.ClassVar = Stats

        def render(self, summaries):
            return {name: s.mean for name, s in summaries.items()}

    result = (Experiment("x", pipeline, frame)
              .scenario("limit_0.3", params=LIMIT_03)
              .metric(Describe(name="tier", column="tier"))
              .plot(Means(metric="tier"))
              .run())
    assert result.plots["test_means"] == {"baseline": 0.875, "limit_0.3": 0.75}
