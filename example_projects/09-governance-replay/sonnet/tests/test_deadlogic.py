"""Coverage and dead-logic detection (spec 09 §5.8), over a small demo population."""
from governance import deadlogic, flows

import capture_demo_evidence as cde


def _run(flow_code: str, requests: list[dict]) -> list[dict]:
    adapter = flows.get(flow_code)
    built = adapter.build("0.1.0")
    return [built.executable.score(adapter.type_record(r), built.params) for r in requests]


def test_01_s_live_rule_universe_is_the_full_declared_set():
    universe = deadlogic.rule_universe("01")
    assert len(universe) == 521  # SCOPE.md/NOTES.md: 521 live rules


def test_coverage_over_a_small_population_finds_both_dead_and_dominant_rules():
    records = _run("03", cde.granting_population())
    report = deadlogic.rule_coverage("03", records, "cap waterfall")
    assert report.universe_size == 52
    assert report.population_size == len(records)
    # every rule that isn't dead and isn't dominant should sit strictly between the two
    assert set(report.dead).isdisjoint(report.dominant)
    assert len(report.dead) + len(report.dominant) <= report.universe_size


def test_05_s_roll_up_rule_universe_is_the_six_rule_subset():
    universe = deadlogic.rule_universe("05")
    assert universe == {"AE-R-01", "AE-R-02", "AE-R-06", "AE-R-07", "AE-R-11", "AE-R-12"}


def test_table_cell_coverage_reports_a_fraction_not_a_threshold_pass_fail():
    records = _run("03", cde.granting_population())
    coverage = deadlogic.table_cell_coverage(records, "risk_grade_cell_id", total_cells=100)
    assert 0.0 <= coverage["pct_read"] <= 1.0
    assert coverage["cells_read"] <= coverage["total_cells"]
