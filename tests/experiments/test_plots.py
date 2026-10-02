from __future__ import annotations

import subprocess
import sys
import typing as t

import polars as pl
import pytest

from decider import branch, flow, param, step
from decider.exceptions import DeciderError
from decider.experiments import (Bars, BranchFlows, Changes, Describe, Divergence, Experiment, ExperimentResult,
                                 Figure, Plot, Sankey, Stats, ValueCounts)
from decider.experiments.plots.sankey import TEMPLATE

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


def _rows(figure: Figure) -> list[dict]:
    return figure.spec["data"]["values"]


def _sweep(pipeline, frame):
    return (Experiment("x", pipeline, frame)
            .scenario("limit_0.3", params=LIMIT_03)
            .metric(Divergence(column="tier"), ValueCounts(column="approved", nodes="demo/approved"))
            .plot(Changes(metric="divergence"), Bars(metric="value_counts"))
            .run())


def test_bars_draws_value_counts_per_scenario_as_an_altair_chart(pipeline, frame):
    bars = _sweep(pipeline, frame).plots["bars"]

    assert bars.grammar == "vega-lite"
    assert bars.spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert {(r["scenario"], r["value"], r["records"]) for r in _rows(bars)} == {
        ("baseline", "True", 3), ("baseline", "False", 1), ("limit_0.3", "True", 2), ("limit_0.3", "False", 2)}


def test_changes_draws_records_changed_per_scenario(pipeline, frame):
    changes = _sweep(pipeline, frame).plots["changes"]
    assert [(r["scenario"], r["changed"]) for r in _rows(changes)] == [("baseline", 0), ("limit_0.3", 1)]


def test_a_custom_plot_returns_an_altair_chart_of_the_metric_it_names(pipeline, frame):
    import altair as alt

    class Means(Plot):
        type: t.Literal["test_means"] = "test_means"
        requires: t.ClassVar = Stats

        def render(self, summaries):
            rows = [{"scenario": name, "mean": s.mean} for name, s in summaries.items()]
            return alt.Chart(alt.Data(values=rows)).mark_point().encode(x="scenario:N", y="mean:Q")

    result = (Experiment("x", pipeline, frame)
              .scenario("limit_0.3", params=LIMIT_03)
              .metric(Describe(name="tier", column="tier"))
              .plot(Means(metric="tier"))
              .run())
    assert [r["mean"] for r in _rows(result.plots["test_means"])] == [0.875, 0.75]


def _nested():
    def is_private(sector_code: int) -> bool:
        return sector_code == 1

    def is_big(size: float) -> bool:
        return size > 60

    def _cap(name, value):
        @step(output="term_cap", name=name)
        def cap(term_cap: float, cap: float = param(value)) -> float:
            return min(term_cap, cap)
        return cap

    inner = branch(is_big, _cap("big", 48.0), _cap("small", 36.0), modifies=["term_cap"], name="by_size")
    return flow(branch(is_private, inner, _cap("public", 60.0), modifies=["term_cap"], name="by_sector"),
                name="demo")


def test_sankey_fills_the_vega_template_one_stack_per_branch_depth_with_the_outcome_last():
    frame = pl.DataFrame({"sector_code": [1, 1, 2], "size": [72.0, 50.0, 72.0], "term_cap": [72.0, 72.0, 72.0]})
    result = (Experiment("branch", _nested(), frame)
              .metric(BranchFlows(outcome="term_cap"))
              .plot(Sankey(metric="branch_flows"))
              .run())
    sankey = result.plots["sankey"]

    assert sankey.grammar == "vega"
    assert sankey.spec["$schema"] == "https://vega.github.io/schema/vega/v5.json"
    assert sankey.spec["marks"] == Figure.of(__import__("json").loads(TEMPLATE.read_text())).spec["marks"]
    values = sankey.spec["data"][0]["values"]
    stacks = {v["label"]: v["stack"] for v in values if "category" in v}
    assert stacks == {"all records": 1, "by_size": 2, "public": 2, "big": 3, "small": 3,
                      "term_cap 36.0": 4, "term_cap 48.0": 4, "term_cap 60.0": 4}
    assert sum(v["value"] for v in values if v.get("source") == "in") == 3


def test_sankey_draws_the_scenario_it_names():
    frame = pl.DataFrame({"sector_code": [1, 2], "size": [72.0, 72.0], "term_cap": [72.0, 72.0]})
    exp = (Experiment("branch", _nested(), frame)
           .scenario("all_public", overrides={"sector_code": 2})
           .metric(BranchFlows(outcome="term_cap"))
           .plot(Sankey(name="public", metric="branch_flows", scenario="all_public")))
    values = exp.run().plots["public"].spec["data"][0]["values"]
    assert {v["label"] for v in values if "category" in v} == {"all records", "public", "term_cap 60.0"}


def test_figures_show_in_a_notebook_through_altairs_renderers(pipeline, frame):
    result = _sweep(pipeline, frame)
    assert "vega-embed" in result.plots["changes"]._repr_mimebundle_()["text/html"]
    vega = Figure(grammar="vega", spec={"$schema": "https://vega.github.io/schema/vega/v5.json", "marks": []})
    assert "vega-embed" in vega._repr_mimebundle_()["text/html"]


def test_result_show_lays_out_status_scenarios_metrics_and_plots(pipeline, frame, monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda *objs: shown.extend(objs))
    result = _sweep(pipeline, frame)
    result.show()

    assert shown[0].data.startswith("### x: completed")
    assert shown[1]["scenario"].to_list() == ["baseline", "limit_0.3"]
    assert shown[5]["counts.True"].to_list() == [3, 2]
    assert shown[-2:] == list(result.plots.values())

    shown.clear()
    result.plots["bars"].show()
    assert shown == [result.plots["bars"]]


def test_a_result_with_plots_round_trips_through_json(pipeline, frame):
    result = _sweep(pipeline, frame)
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored.plots == result.plots


def test_core_decider_and_the_metrics_never_import_altair():
    code = ("import sys, decider, decider.experiments, decider.mcp.tools; "
            "assert 'altair' not in sys.modules and 'IPython' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_plot_without_altair_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "altair", None)
    with pytest.raises(DeciderError, match=r"decider\[experiments\]"):
        Changes(metric="divergence").render({})
