"""`attest()` - the combinator a flow wraps itself in, and the reason this
project can be a property of the estate rather than a per-team chore.

THE DEVIATION FROM DOC 03
-------------------------
Doc 03 §7 offers `taps=["term_cap", "branch_path"]` as the diagnostic surface:
per-module, opt-in, author-chosen. That is exactly the wrong shape for spec 09.
§5.15 items 14, 15 and 16 demand that EVERY branch evaluation, EVERY cap chain
and EVERY score contribution be recorded, on every flow, forever. A per-module
opt-in list means some team, one quarter, forgets one module - and the gap is
found by an auditor in 2033, retrospectively, permanently.

So: taps stay, for the case they are good at (a hand-picked intermediate an
engineer wants in a dashboard). Evidence does not go through them.

    TermLoan = attest(TermLoanPipeline, under=CONTRACT_V3)

`attest()` is a combinator in the doc 02 §1.2 sense - data in the graph, it
renders, diffs and serialises - with three properties:

1. It is SEMANTICALLY TRANSPARENT, like `fuse()`. It changes codegen and
   nothing else, and `testing/equivalence.py` asserts
   `attest(P) == P` on the certification corpus. A governance annotation that
   could change an answer would be the worst possible thing to put on 22 M
   decisions a year.
2. It DERIVES the witness set from the graph, not from a list. Every value
   overwritten at more than one module boundary is chained; every Branch emits
   its `branch_path`; every table read emits `(version, flat_index)`. Nobody
   types a name, so nobody forgets one.
3. It REFUSES TO BUILD when the contract fails structurally. This is the
   single most valuable line in the project: non-compliance is a compile
   error in the flow's own CI, not a finding in the harness's monthly report.

The cost is stated honestly in FRAMEWORK-DEMANDS D3: an attested kernel is
wider than an unattested one, the witness columns are the dominant write-back
cost at high output counts (doc 02 §1 - write-back is already 54.7% at 633
outputs), and §8.3's storage tiering exists because of it.
"""

from __future__ import annotations

from decider2 import Pipeline, combinator
from decider2.graph import Module

from contract.items import ITEMS, Proof


@combinator
def attest(pipeline: Pipeline, *, under: "Contract", emit: "WitnessPolicy") -> Pipeline:
    """Wrap a pipeline so that it emits a §5.15-complete witness on every call."""
    pass  # static-check the contract, derive the witness set from the graph, return an annotated pipeline


class WitnessPolicy:
    """Answers spec 09 §8.3's five-times storage question explicitly, in code,
    rather than by whichever default the implementation happened to have."""

    always: tuple[str, ...]      # the §5.15 fields. 3.1 KB/decision. Never sampled.
    full_detail_when: "Predicate"  # declines | referrals | disputed | deterministic 0.5% sample
    regenerate_on_demand: bool   # everything else comes back from replay, which is why replay must be exact


# The decision, for the credit flows, stated once:
CREDIT_WITNESS = WitnessPolicy(
    always=("inputs_as_received", "reference_versions", "params_resolved", "structure_fingerprint",
            "capability_versions", "assignments", "decision_date", "overlay_stack",
            "branch_paths", "cap_chains", "score_contributions", "reason_codes"),
    full_detail_when=lambda d: (d.outcome_code in (2, 3)          # refer, decline
                                or d.offered_below_requested      # a reduction is a statutory-reason event
                                or d.sample_flag                  # deterministic 0.5%, hashed, never random
                                or d.disputed),
    regenerate_on_demand=True,
)


# ---------------------------------------------------------------------------
# `holds` - the second invention, and the answer to doc 04 §6.2 and §13 Q10.
# ---------------------------------------------------------------------------
#
# Doc 04 §6 proposes generating the reviewable artefact from module data plus
# the docstring. A docstring is prose nobody checks. It rots in weeks, exactly
# as spec 09 §5.7 says a hand-maintained policy document rots, and generating
# it does not fix that - it only moves the rot inside the generated artefact,
# where it is harder to see.
#
# A `holds` is prose that CI runs.
#
#     @holds("below R5 000 net salary no term above 48 months is ever offered",
#            given={"min_net_salary": 499_900}, then=lambda r: r.term_cap <= 48)
#     @holds("at or above the floor the rule never reduces the term",
#            given={"min_net_salary": 500_100, "term_cap": 72}, then=lambda r: r.term_cap == 72)
#     def cap_by_income_band(term_cap: float, min_net_salary: float,
#                            cap: float = param(48.0, ge=6, le=60)) -> float:
#         ...
#
# It does two jobs from one artefact, which is doc 01 §5.3's law applied to
# governance: the reviewable artefact prints the SENTENCE, and the
# certification artefact prints "verified by 2 assertions, build 7c41e9, both
# passing". The reviewer is then not verifying the logic - they are verifying
# that the claims are the right claims, which is a thing a non-engineer can
# actually do.
#
# And the honesty mechanism, without which this is decoration: a `holds` that
# still passes after the rule it describes is MUTATED is not describing it.
# `certify.mutation_screen()` perturbs each rule's params and bounds and flags
# every assertion that survives. A surviving assertion is reported as
# NON-DISCRIMINATING and the clause is rendered with that word on it.

def holds(sentence: str, *, given: dict, then: "Callable[[Values], bool]",
          approved: str | None = None) -> "Decorator":
    """Attach a reviewable, executable claim to a step or module."""
    pass  # register the claim on the step; certification runs it; the renderer prints `sentence`
