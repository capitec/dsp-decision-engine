"""Swap-set analysis and attribution (spec 09 §5.5, acceptance §10 item 20)."""
import capture_demo_evidence as cde
from governance import overlay_register, swapset


def test_attributed_swap_set_isolates_the_overlay_increment():
    """Disabling the overlay stack is a pure "overlay" cause with nothing else changed
    (09 §5.14.4): the swap set for that one increment must be attributable to it alone,
    against the run immediately before it."""
    population = cde.granting_population()

    def disable_overlay_stack(params):
        return overlay_register._set_every(params, "adjustment_stack_enabled", False)

    reports = swapset.attributed_swap_set("03", population, [
        ("disable overlay stack", disable_overlay_stack),
    ])
    assert len(reports) == 2  # baseline + one increment, per §5.5's "n changes -> n+1 runs"
    baseline, overlay_off = reports
    assert baseline.label == "baseline (no change)"
    assert baseline.outcome_moved_count == 0  # comparing the baseline against itself
    assert overlay_off.population_size == len(population)


def test_two_version_swap_set_reports_unmatched_and_moved_separately():
    population = cde.granting_population()[:3]
    report = swapset.swap_set("03", population, "0.1.0", "0.1.0", label="same version")
    # same version both sides: nothing should move, nothing should be unmatched
    assert report.outcome_moved_count == 0
    assert report.unmatched == ()
    assert report.population_size == len(population)
