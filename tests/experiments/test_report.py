from __future__ import annotations

import pytest

from decider.experiments import Experiment, ExperimentDef, validate
from decider.experiments.metrics import Divergence, ValueCounts
from decider.experiments.plots import Changes
from decider.experiments.report import Chart, MetricSection, Overview, Report, Row, Section, Table, Text

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


def _exp(pipeline, frame):
    return (Experiment("x", pipeline, frame)
            .scenario("limit_0.3", params=LIMIT_03)
            .metric(Divergence(column="tier"), ValueCounts(column="approved", nodes="demo/approved"))
            .plot(Changes(metric="divergence")))


def _order(html: str, *parts: str) -> list[int]:
    return [html.index(p) for p in parts]


def test_the_default_report_is_an_overview_then_a_section_per_metric(pipeline, frame):
    html = _exp(pipeline, frame).run().html()

    assert html.startswith("<!DOCTYPE html>")
    first, second = _order(html, "<h3>divergence</h3>", "<h3>value_counts</h3>")
    assert html.index("completed</span></h2>") < first < second
    assert "at steps matching <code>demo/approved</code>" in html
    assert html.count("embed(el,") == 2  # divergence's own plot, and the built-in Bars for value_counts


def test_a_report_lays_out_text_tables_and_charts_where_it_says(pipeline, frame):
    result = (_exp(pipeline, frame)
              .report(Text(markdown="## Why\nThe *limit* moved."),
                      Section(title="Changes", blocks=[Row(blocks=[Table(metric="divergence"), Chart(plot="divergence")])]),
                      MetricSection(metric="value_counts"),
                      title="Limit report")
              .run())
    html = result.html()

    assert "<title>Limit report</title>" in html
    positions = _order(html, "<h1>Limit report</h1>", "<em>limit</em>", "<h2>Changes</h2>", '<div class="row">',
                       "<h3>value_counts</h3>")
    assert positions == sorted(positions)
    assert "Overview" not in html and "<h2>x " not in html  # only what the report asks for


def test_markdown_in_a_report_cannot_inject_html(pipeline, frame):
    html = _exp(pipeline, frame).report(Text(markdown="<script>alert(1)</script>")).run().html()
    assert "<script>alert(1)" not in html and "&lt;script&gt;alert(1)" in html


def test_a_report_must_name_metrics_and_plots_the_experiment_has(pipeline, frame):
    exp = _exp(pipeline, frame).report(Overview(), Section(title="s", blocks=[Chart(plot="nope"),
                                                                              Row(blocks=[Table(metric="nah")])]))
    assert sorted(validate(exp.def_, pipeline)) == [
        "the report shows metric 'nah', which the experiment doesn't have",
        "the report shows plot 'nope', which the experiment doesn't have"]
    with pytest.raises(ValueError, match="nope"):
        exp.run()


def test_a_report_round_trips_through_json_with_the_experiment(pipeline, frame, tmp_path):
    exp = _exp(pipeline, frame).report(Overview(), Row(blocks=[Chart(plot="divergence"), Text(markdown="hi")]))
    restored = ExperimentDef.model_validate_json(exp.save(tmp_path / "experiment.json").read_text())
    assert restored.report == exp.def_.report
    assert isinstance(restored.report.blocks[1], Row) and isinstance(restored.report.blocks[1].blocks[0], Chart)


def test_save_html_writes_the_page_and_show_draws_it_in_a_notebook(pipeline, frame, tmp_path, monkeypatch):
    import IPython.display

    result = _exp(pipeline, frame).run()
    saved = result.save_html(tmp_path / "report.html").read_text()
    assert saved.startswith("<!DOCTYPE html>") and "<h3>divergence</h3>" in saved

    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda *objs: shown.extend(objs))
    result.show()
    assert len(shown) == 1 and shown[0].data.startswith('<div class="decider-report">')


def test_a_report_without_a_title_still_names_the_page(pipeline, frame):
    assert "<title>x</title>" in _exp(pipeline, frame).run().html()
    assert Report().blocks == []
