from __future__ import annotations

import subprocess
import sys
import typing as t

import polars as pl
import pytest

from decikit import branch, flow, param, step
from decikit.exceptions import DecikitError
from decikit.lab.experiments import Experiment, ExperimentResult
from decikit.lab.experiments.metrics import BranchFlows, Counts, Describe, Divergence, Histogram, Stats, ValueCounts
from decikit.lab.experiments.plots import Bars, Changes, Distribution, Figure, Plot, Ranges, Sankey
from decikit.lab.experiments.plots._sankey import TEMPLATE

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
    bars = _sweep(pipeline, frame).plots["value_counts"]

    assert bars.grammar == "vega-lite"
    assert bars.spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert {(r["scenario"], r["value"], r["records"]) for r in _rows(bars)} == {
        ("baseline", "True", 3), ("baseline", "False", 1), ("limit_0.3", "True", 2), ("limit_0.3", "False", 2)}


def test_changes_draws_records_changed_per_scenario(pipeline, frame):
    changes = _sweep(pipeline, frame).plots["divergence"]
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
    assert [r["mean"] for r in _rows(result.plots["tier"])] == [0.875, 0.75]


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
    sankey = result.plots["branch_flows"]

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
    assert "embed(el," in result.plots["divergence"]._repr_mimebundle_()["text/html"]
    vega = Figure(grammar="vega", spec={"$schema": "https://vega.github.io/schema/vega/v5.json", "marks": []})
    assert 'mode: "vega"' in vega._repr_mimebundle_()["text/html"]


def test_distribution_overlays_each_scenarios_histogram_with_overflow_at_the_ends():
    from example_projects.experiments.loan_pipeline import applications, build

    result = (Experiment("x", build(), applications(2_000))
              .scenario("premium_6pct", params={"loans": {"pricing": {"unsecured": {"premium": 0.06}}}})
              .metric(Histogram(column="rate", nodes="loans/pricing/unsecured", bins=10))
              .plot(Distribution(metric="histogram"))
              .run())
    rows = _rows(result.plots["histogram"])
    shifted = result.summaries["histogram"]["premium_6pct"]
    assert shifted.above > 0
    assert sum(r["records"] for r in rows if r["scenario"] == "premium_6pct") == sum(shifted.counts) + shifted.above
    assert result.plots["histogram"].spec["params"][0]["bind"] == "legend"


def _bins_summaries():
    from decikit.lab.experiments.metrics import Bins
    return {name: Bins(column="rate", edges=[0.0, 0.5, 1.0], counts=counts, below=0, above=0, nulls=0)
            for name, counts in (("baseline", [3, 1]), ("faster", [1, 3]))}


def test_distribution_layouts_place_scenarios_over_beside_on_top_or_in_rows():
    def encoding(layout):
        spec = Figure.of(Distribution(metric="m", layout=layout).render(_bins_summaries())).spec
        return spec.get("spec", spec)["encoding"], spec

    overlay, _ = encoding("overlay")
    assert overlay["y"].get("stack", "missing") is None and overlay["x"]["bin"] == "binned"
    assert encoding("stacked")[0]["y"]["stack"] == "zero"
    beside, _ = encoding("side_by_side")
    assert beside["xOffset"]["field"] == "scenario" and beside["x"]["sort"] == ["0–0.5", "0.5–1"]
    _, rows = encoding("rows")
    assert rows["facet"]["row"]["field"] == "scenario"


def test_bars_layouts_and_a_plots_size():
    def spec(**kwargs):
        summaries = {"a": Counts(column="x", counts={"T": 1}), "b": Counts(column="x", counts={"T": 2})}
        return Figure.of(Bars(metric="m", **kwargs).render(summaries)).spec

    assert "xOffset" in spec()["encoding"] and "xOffset" not in spec(layout="stacked")["encoding"]
    assert spec(layout="rows")["facet"]["row"]["field"] == "scenario"
    assert (spec(width=500, height=120)["width"], spec(width=500, height=120)["height"]) == (500, 120)


def test_changes_draws_one_horizontal_bar_per_scenario(pipeline, frame):
    changes = _sweep(pipeline, frame).plots["divergence"].spec
    assert changes["encoding"]["y"]["field"] == "scenario" and changes["encoding"]["x"]["field"] == "changed"


def test_every_figure_on_a_page_shares_one_load_of_vega_and_escapes_its_spec():
    html = Figure(grammar="vega", spec={"title": "</script><b>x</b>"}).html()
    assert html.count("</script>") == 1 and "<\\/script>" in html
    assert "window.decikitVega = window.decikitVega ||" in html


def test_ranges_draws_min_mean_and_max_per_scenario(pipeline, frame):
    result = (Experiment("x", pipeline, frame).scenario("limit_0.3", params=LIMIT_03)
              .metric(Describe(column="tier")).plot(Ranges(metric="describe")).run())
    ranges = result.plots["describe"]
    marks = [layer["mark"] for layer in ranges.spec["layer"]]
    assert [m["type"] if isinstance(m, dict) else m for m in marks] == ["rule", "point"]
    assert "embed(el," in ranges._repr_mimebundle_()["text/html"]  # a layered chart shows as one too


def test_a_result_with_plots_round_trips_through_json(pipeline, frame):
    result = _sweep(pipeline, frame)
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored.plots == result.plots


def test_core_decikit_and_the_metrics_never_import_altair():
    code = ("import sys, decikit, decikit.lab.experiments, decikit._tooling.mcp._tools; "
            "assert 'altair' not in sys.modules and 'IPython' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_plot_without_altair_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "altair", None)
    with pytest.raises(DecikitError, match=r"decikit\[experiments\]"):
        Changes(metric="divergence").render({})
