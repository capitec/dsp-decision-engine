"""Release certification, and the only test in this project that blocks.

Spec §5.6, and the non-functional constraint that shapes everything: the full
suite across eight flows, 400 000 records, in a 40-MINUTE CI budget; one flow's
50 000 records in under 6 minutes locally. A certification gate that takes two
hours becomes a gate that is bypassed.

THE BUDGET IS MET BY THE FRAME TIER AND BY NOTHING ELSE
  50 000 records through a flow-03-shaped pipeline is ~12 kernels over 50 000
  rows. At the measured 0.20 ns/step/row for split kernels (doc 02 §1.1) the
  arithmetic is trivially inside the budget; the cost is the iterative solve
  (flow 03 searches a money grid per application) and the comparison, not the
  logic. Two consequences the framework docs make available and nobody has
  connected to certification:

  - `apply()` never fuses across module boundaries by default, and that is the
    RIGHT default here because 50 000 rows is above the ~10k break-even where
    fusion turns harmful. A team that wrapped their whole flow in `fuse()` for
    a realtime win would pay 3-5x in certification. `explain_kernels()` in the
    certification artefact makes that visible.
  - The golden comparison is a polars join and a set of per-column tolerance
    predicates. It parallelises across flows and not within a golden set, which
    is exactly what spec §5.6 says, and the reason is the join.

SCOPED RE-CERTIFICATION - "a change to one phase not requiring re-certification
of everything" - IS STATIC LINEAGE DOING A JOB NOBODY ASKED IT TO DO
  `manifest.cone(changed_modules)` gives the outputs reachable from a change,
  with no execution. Intersect with the coverage index (which golden records
  exercised a branch inside those modules) and you have the re-certification
  scope. For the worked release: increment 3 touches `cap_register` only, whose
  cone reaches amount_cap, term_cap, worst_acceptable_grade and everything
  downstream of them - which is most of the flow - but the COVERAGE
  intersection is 8 140 of 50 000 records, so the local run is 58 seconds
  instead of 5 minutes.

  The honest caveat, and it is large: a change inside a module that produces
  `risk_grade` has a cone of the entire flow, and the coverage intersection is
  every record. Scoped re-certification saves time on leaf-ish changes and
  saves nothing on central ones. It is not a general answer; it is an answer
  for the 60% of changes that are peripheral.
"""

from __future__ import annotations

from release.increment import Increment, NoOutcomeChange, Release


class ToleranceClass:
    """From harness/tolerances.py - versioned, effective-dated, Credit
    Committee-approved, because a change to a tolerance band is a governance
    change (spec §9.2)."""
    name: str
    rule: str


TOLERANCES = {
    "outcome_code": "exact",
    "decline_reason_codes": "exact, including order",
    "primary_reason_code": "exact",
    "tree_node_path": "exact",
    "money_after_rounding": "exact to the cent",
    "rates": "exact to 4 dp",
    "scores_probabilities": "1e-6 absolute",
    "intermediates_unrounded": "1e-12 relative",
}


class GoldenSet:
    """50 000 per flow. Stratified, not random: products, grades, channels,
    segments, outcome types, ~14 strata. Deliberately loaded with band edges,
    rounding boundaries, exactly-at-threshold values, maximum-cardinality
    collections and every null pattern seen in production.

    Two rules that are the whole value of the set:
      - refreshed quarterly with newly-seen edge cases;
      - EVERY RECORD THAT HAS EVER CAUGHT A DEFECT IS RETAINED FOREVER.

    And doc 03 §1.2's warning, which is a requirement on this set specifically:
    the int64 overflow at realistic loan sizes was found by binary search, not
    by sampling. Random draws would never surface it. So a stratum of the
    golden set is generated adversarially from the manifest - every band edge
    of every table the flow reads, +/- 1 cent - rather than sampled from
    production at all.
    """
    flow: str
    version: str
    records: int
    masked: bool = True          # spec §5.13.3 - non-production use is masked BY DEFAULT
    strata: dict[str, int]


class CertificationResult:
    release: Release
    verdict: "Literal['certified', 'blocked']"
    unexpected_moves: tuple[str, ...]        # moved, not declared to        -> blocks
    absent_moves: tuple[str, ...]            # declared to move, did not     -> blocks
    wrong_direction: tuple[str, ...]         # moved wrongly or too far      -> blocks
    coverage: "CoverageResult"
    claims: "ClaimResult"
    modes_agree: bool                        # assert_modes_agree over the golden set, per build
    elapsed_seconds: float


def certify(release: Release, *, golden: GoldenSet) -> CertificationResult:
    """Per-increment, then cumulative. The per-increment result is what makes
    `NoOutcomeChange` enforceable: a refactor buried inside a release that also
    moves a threshold is otherwise indistinguishable from the threshold move."""
    pass


def mutation_screen(release: Release, *, golden: GoldenSet) -> "ClaimResult":
    """Spec §13 Q10 - how is the description that makes a rendering readable
    kept honest?

    Every `holds` sentence on every rule the release touches is re-run against
    a MUTATED version of that rule: each param perturbed to its declared
    bounds, each comparison operator flipped, each set membership negated. An
    assertion that still passes is NON-DISCRIMINATING - it does not describe
    the rule it is attached to - and the reviewable artefact renders that
    clause with the words 'claim not verified by test' in place of the usual
    'verified by N assertions'.

    This is the only mechanism in the project that stops the reviewable
    artefact decaying into plausible prose, and it costs one extra golden run
    per touched rule (the screen runs on the ~8 000-record coverage
    intersection, not the full set).
    """
    pass


def coverage_thresholds() -> dict:
    return {
        "rules_exercised": 0.98,
        "tree_nodes_reached": 0.95,
        "reason_codes_reachable": 1.00,
        "touched_paths_distinguished": 1.00,   # every code path a release touches exercised by at
                                               # least one record that distinguishes it from the
                                               # previous version - this is the one that catches
                                               # "we tested it and the test would have passed either way"
        # table cells: MEASURED, NOT THRESHOLDED. 50 000 records cannot cover
        # 219 600 cells and pretending otherwise produces a meaningless number.
    }
