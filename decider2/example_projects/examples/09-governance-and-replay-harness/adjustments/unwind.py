"""Running a flow with its overlay stack disabled, through the SAME
implementation - and the estimated cost of unwinding each expired overlay.

Spec §5.14.3 is explicit that a separate "unadjusted" calculation is the same
defect as a separate simulation implementation in flow 07 and a separate
backtest implementation in flow 01: it will agree at first and diverge silently,
and the divergence will be discovered during the exercise that most needed it
to be right.

HOW ONE IMPLEMENTATION RUNS BOTH WAYS, IN decider2 TERMS
  An overlay is not a branch in the flow's graph. It is a VALUE that
  `core.adjustments` applies, and the set of overlays in force arrives as part
  of the resolved params bundle for the module that consumes it. So:

      base = flow03.at_pin(pins.with_overlays(DISABLED))

  ...is a PARAMS SWAP (doc 08 §4.2 - free, no compile, microseconds), not a
  different pipeline, not a flag inside a step, not a second code path. The
  compiled kernel is byte-identical. The equivalence is not asserted; there is
  nothing to assert, because there is one kernel.

  That only works if the overlay set's TYPE is fixed across enabled/disabled -
  which it is, if a disabled overlay is a magnitude of zero rather than an
  absent entry. An absent entry changes a container field's length, and doc 08
  §4.2 lists exactly that as a recompile trigger. FRAMEWORK-DEMANDS D11: the
  overlay bundle must be a fixed-width array over the register version, with
  `enabled` as a mask, or "run with the stack off" silently becomes a compile
  on the request path.

COST SHAPE (spec §5.14.3)
  Running every flow both ways for every decision is not affordable at 22 M
  decisions a year - flow 03 alone is 55 000 applications a day through a
  search that dominates its cost. So:
    - the INPUTS to each overlay (`score_unadjusted`,
      `probability_of_default_unadjusted`, the unadjusted grade, the unadjusted
      cap chain, the card cell before the add-on) are recorded on EVERY
      decision, including declines, because they are cheap (item 7, acceptance
      criterion 21);
    - the COUNTERFACTUAL OUTCOME is produced on demand, over samples for
      monitoring and over full populations for impact estimation, by the same
      implementation.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from replay.pin_resolution import PinSet


def disabled(pins: PinSet) -> PinSet:
    """Same pin, overlay magnitudes masked to zero. Fixed-width, same type."""
    pass


def unwind_estimate(overlay_id: str, *, month: str) -> "UnwindEstimate":
    """Spec §5.14.2's row with teeth: what would removing this overlay do?

    Run the month's population with the full stack, then with this one overlay
    masked, and report the swap set. Note what this is NOT: it is not the
    overlay's original effect estimate, because the world moved - the
    population changed, the base model drifted, other overlays stacked on top.
    'What it would cost to remove it today' and 'what it bought when it was
    approved' are different numbers and both are reported, because a committee
    comparing them is the point.

    Compute: one masked run per overlay per month over a 200 000-record
    stratified sample rather than 2 M, because the estimate needs two
    significant figures and not four. 90 live overlays x one sample run is
    ~4 hours monthly, which fits. At full population it would be ~45 hours,
    which does not, and the report would then be produced quarterly, which is
    how expiry stops having teeth.
    """
    pass


def model_base_predictions(flow: str, month: str) -> pl.LazyFrame:
    """Spec §11.13. Model Risk rejected a validation because the monitoring pack
    compared realised defaults against ADJUSTED predictions - measuring the
    overlay and reporting it as model accuracy. The validation opinion covers
    the model, not the policy overlay laid over it.

    This produces base-model predictions on the production population,
    INCLUDING where an overlay caused a decline (so the population is the
    applicants, not the approvals), retrospectively for any month in the
    retention window - which is only possible because item 7 recorded
    `score_unadjusted` and `probability_of_default_unadjusted` on every
    decision, including the declines, at the time.
    """
    pass


def absorb(overlay_id: str, into: str) -> "AbsorptionPlan":
    """Spec §11.16-17: moving a dominant overlay into the base artefact without
    breaking the comparability of two years of decisions.

    The plan: the base artefact gets a new version whose cells or boundaries
    equal base+overlay; the overlay is retired with an explicit `absorbed_into`
    reference; and - the part that preserves comparability - every decision
    made under the old pair keeps its recorded stack, while the register
    publishes an EQUIVALENCE: 'grade boundary 6/7 at PD 0.0470 in RGB-2027-05
    equals RGB-2026-11 plus ADJ-2026-052'. Analyses that span the boundary read
    the equivalence rather than silently comparing incomparable things.

    Scenario 17's prohibition (spec §11.17 - a flow merging an overlay into a
    base table during a refactor 'to simplify') is the same operation done
    without this plan, and the harness detects it structurally: a base-artefact version whose
    cells moved by exactly a live overlay's magnitude over exactly its scope,
    with no absorption plan, is flagged at diff time.
    """
    pass
