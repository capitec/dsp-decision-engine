from __future__ import annotations

import typing as t

import polars as pl

from decider import branch, flow, param, step
from decider.experiments import (Batch, BranchFlows, Describe, Divergence, Experiment, ExperimentDef, Histogram,
                                 Metric, Sankey, Summary)

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


def _batch(output, baseline, offset=0):
    return Batch("s", 0, offset, pl.DataFrame({"x": output}), pl.DataFrame({"x": baseline}))


def _divergence(output, baseline, **kwargs):
    m = Divergence(column="x", **kwargs)
    m.reset()
    m.on_batch_end(_batch(output, baseline))
    return m.summarise()


def test_divergence_reports_changed_records_the_first_one_and_samples(pipeline, frame):
    result = Experiment("x", pipeline, frame).scenario("limit_0.3", params=LIMIT_03).metric(
        Divergence(column="tier")).run()

    assert result.summaries["divergence"]["baseline"].changed == 0
    variant = result.summaries["divergence"]["limit_0.3"]
    assert (variant.compared, variant.changed, variant.first) == (4, 1, 2)
    assert variant.samples == [{"row": 2, "expected": 1.0, "actual": 0.5}]


def test_divergence_holds_floats_to_tolerance_and_everything_else_exactly():
    assert _divergence([1.0 + 1e-12, float("nan")], [1.0, float("nan")]).changed == 0
    assert _divergence([1.1], [1.0], rtol=0.2).changed == 0
    assert _divergence([1.1], [1.0]).changed == 1
    assert _divergence(["a", "b"], ["a", "c"]).first == 1


def test_divergence_counts_null_as_equal_only_to_null():
    assert _divergence([None, None, 1.0], [None, 1.0, None]).changed == 2


def test_divergence_numbers_records_across_batches():
    m = Divergence(column="x", samples=1)
    m.reset()
    m.on_batch_end(_batch([1, 2], [1, 2]))
    m.on_batch_end(_batch([3, 9, 9], [3, 4, 5], offset=2))
    s = m.summarise()
    assert (s.compared, s.changed, s.first, s.samples) == (5, 2, 3, [{"row": 3, "expected": 4, "actual": 9}])


def test_describe_accumulates_over_batches(pipeline, frame):
    result = Experiment("x", pipeline, frame, batch_size=3).metric(Describe(column="tier")).run()
    stats = result.summaries["describe"]["baseline"]
    assert (stats.count, stats.nulls, stats.mean, stats.min, stats.max) == (4, 0, 0.875, 0.5, 1.0)


def test_a_node_metric_sees_only_matching_nodes_with_the_baseline_alongside(pipeline, frame):
    seen = []

    class Watch(Metric):
        type: t.Literal["test_watch"] = "test_watch"

        def on_node_end(self, node):
            seen.append((node.scenario, node.path, node.values.columns, node.baseline is not None))

        def summarise(self):
            return Summary()

    Experiment("x", pipeline, frame).scenario("limit_0.3", params=LIMIT_03).metric(Watch(nodes="demo/a*")).run()
    assert seen == [("baseline", "demo/approved", ["approved"], True),
                    ("limit_0.3", "demo/approved", ["approved"], True)]


def test_a_global_metric_never_gets_node_hooks(pipeline, frame):
    class Nodes(Metric):
        type: t.Literal["test_nodes"] = "test_nodes"

        def reset(self):
            self._n = 0

        def on_node_end(self, node):
            self._n += 1

        def summarise(self):
            return Summary(n=self._n)

    result = Experiment("x", pipeline, frame).metric(Nodes()).run()
    assert result.summaries["test_nodes"]["baseline"].n == 0


class Total(Summary):
    records: int


class RecordCount(Metric):
    type: t.Literal["test_record_count"] = "test_record_count"
    summary: t.ClassVar = Total

    def reset(self):
        self._n = 0

    def on_batch_end(self, batch):
        self._n += batch.output.height

    def summarise(self):
        return Total(records=self._n)


def test_a_custom_metric_loads_from_json_by_its_alias_and_summarises_per_scenario(pipeline, frame):
    def_ = ExperimentDef.model_validate_json(
        '{"name": "x", "flow": {"entry": "demo:build"}, "batch_size": 3,'
        ' "scenarios": [{"name": "limit_0.3"}], "metrics": [{"type": "test_record_count"}]}')
    exp = Experiment("x", pipeline, frame)
    exp.def_ = def_

    summaries = exp.run().summaries["test_record_count"]
    assert summaries == {"baseline": Total(records=4), "limit_0.3": Total(records=4)}


def test_branch_flows_counts_every_record_once_through_arms_and_outcomes():
    def is_private(sector_code: int) -> bool:
        return sector_code == 1

    @step(output="term_cap")
    def cap_a(term_cap: float, cap: float = param(54.0)) -> float:
        return min(term_cap, cap)

    @step(output="term_cap")
    def cap_b(term_cap: float, cap: float = param(60.0)) -> float:
        return min(term_cap, cap)

    pipeline = flow(branch(is_private, cap_a, cap_b, modifies=["term_cap"], name="by_sector"), name="demo")
    frame = pl.DataFrame({"sector_code": [1, 2, 2], "term_cap": [72.0, 72.0, 72.0]})
    result = (Experiment("branch", pipeline, frame, batch_size=2)
              .metric(BranchFlows(outcome="term_cap"))
              .plot(Sankey(metric="branch_flows"))
              .run())

    flows = result.summaries["branch_flows"]["baseline"]
    links = {(link["source"], link["target"]): link["value"] for link in flows.links}
    assert links == {("in", "demo/by_sector#0"): 1, ("in", "demo/by_sector#1"): 2,
                     ("demo/by_sector#0", "out:54.0"): 1, ("demo/by_sector#1", "out:60.0"): 2}
    assert [n["label"] for n in flows.nodes if n["kind"] == "arm"] == ["cap_a", "cap_b"]
    assert result.plots["branch_flows"].grammar == "vega"


def _histogram(batches, **kwargs):
    m = Histogram(column="x", **kwargs)
    m.reset()
    for i, (output, baseline) in enumerate(batches):
        m.on_batch_end(_batch(output, baseline, offset=i * 10))
    return m.summarise()


def test_histogram_bins_span_the_baselines_first_batch_for_every_scenario():
    bins = _histogram([([0.0, 5.0, 10.0, 99.0, None], [0.0, 10.0, 5.0, 5.0, 5.0]), ([-1.0, 2.0], [0.0, 0.0])], bins=2)
    assert bins.edges == [0.0, 5.0, 10.0]
    assert (bins.counts, bins.below, bins.above, bins.nulls) == ([2, 2], 1, 1, 1)


def test_histogram_range_and_edges_fix_the_bins_up_front():
    assert _histogram([([0.5, 1.5], [9.0, 9.0])], bins=4, range=(0.0, 2.0)).counts == [0, 1, 0, 1]
    assert _histogram([([0.5, 1.5], [9.0, 9.0])], edges=[0.0, 1.0, 3.0]).counts == [1, 1]


def test_a_metric_on_a_branch_sees_what_the_branch_passes_on_for_every_record():
    from example_projects.experiments.loan_pipeline import applications, build

    frame = applications(1_000)
    result = (Experiment("x", build(), frame)
              .metric(Describe(name="merged", column="rate", nodes="loans/pricing"),
                      Describe(name="unsecured", column="rate", nodes="loans/pricing/unsecured"),
                      Describe(name="before", column="rate", nodes="loans/rate"))
              .run())
    merged, unsecured, before = (result.summaries[k]["baseline"] for k in ("merged", "unsecured", "before"))
    assert merged.count == before.count == 1_000
    assert unsecured.count == (frame["collateral"] == 0).sum()
    assert unsecured.min > before.min
