"""O17 -- authority routing. Seven levels, and the level moves while you decide.

Spec 5.14, H7. The part that is routinely missed:

    The authority level is determined by the PROPOSED structure, and the
    proposed structure changes while O13 is structuring it.

    (c) the analyst adds a surety to improve the price; cover rises; security
        type improves; the level drops to the analyst's own -- AND THE PERSON
        WHO STRUCTURED IT CAN NOW APPROVE IT.

That is not misconduct. It is the natural consequence of a matrix keyed on
something the assessor controls, and a design that treats it as a control
failure will build a detective control for a structural problem.
"""

from decider2 import module, step, param, Loop, approval_binds_to
from modules.security.allocation import authority_of_the_affected_set

# 9 amount bands x 12 grades x 4 security types x 9 facility types = 3 888 cells.
authority_matrix = param.table("delegated_authority", rows=3_888,
                               owner="credit_governance", approved_by="board_credit_committee",
                               cadence="annual, and on mandate change")


def base_level(group_exposure_after_cents: int, risk_grade: int,
               security_type: int, facility_type: int) -> int:
    """The amount dimension is TOTAL GROUP EXPOSURE AFTER THE PROPOSAL, not the
    facility amount. Spec 5.14.1.

    "A R400 000 overdraft to a client whose group already carries R38 000 000 is
     a level 5 decision, and a matrix keyed on the facility amount would route it
     to level 2."

    The input name is therefore `group_exposure_after_cents` and there is no
    `facility_amount_cents` in this module's interface at all. A name that is
    not in scope cannot be read by mistake; a name that is in scope and wrong
    will be read by somebody in year three.
    """
    pass  # 4-D matrix lookup


def modifiers(base_level: int, state: dict) -> int:
    """Twelve modifiers that raise the level irrespective of the cell. Spec 5.14.1.

    watchlist W2 (+1) or W3+ (level 6 min); live forbearance (+1); a second
    concession in probation (level 6); probable screening match (level 6);
    structure_unresolved (level 6); restricted sector (+1); approve at grade
    10-12 (level 6); a waiver on a covenant already waived twice (level 6);
    any policy exception (+1, +2 against an absolute rule); group exposure above
    the single-name limit (level 7); a degraded review at basis 4 or 5 (+1);
    a decision affecting a facility other than the subject (the authority is
    that of the whole affected set); connected-party lending (level 7, any
    amount).

    Four of these are the STATE FLAGS loaded at O1 (spec 5.3 O1). They are
    inputs to routing, which means O1 has to run before O17 can be computed at
    all -- and EP-9, which routes to nobody, must still record that it routed to
    nobody. `routed_to=NOBODY` is a value, not an absence.
    """
    pass  # apply in declared order; record every modifier that fired


def required_level(base_level: int, modifiers: int, affected_set: list) -> int:
    """max(cell + modifiers, authority of the whole affected set)."""
    pass  # authority_of_the_affected_set(affected_set) folded in


# --------------------------------------------------------------------------
# The level moves during the search. Spec 5.14.3, requirement 1.
#
# O13's structuring search is a `Loop` over candidates (05 5.12, consumed). The
# authority is CARRIED through it, so:
#   - it is recomputed on every change to the proposed structure, by construction;
#   - the SEQUENCE of levels the proposal passed through is a carried history,
#     which is spec 5.14.3's "record the sequence" without a separate log;
#   - and it is a second source of non-monotonicity in the search, on top of the
#     two project 05 already names. Reducing the amount can improve the cover
#     ratio, move security_type 2 -> 1, lower the rate, improve the DSCR, and
#     raise the admissible amount again -- while the approver changes twice.
#
# `carries=` is doc 03 8.3, used for exactly what it is for. Nothing new is
# needed here, which is worth saying: the framework's existing Loop handles the
# hardest-sounding requirement in spec 5.14 unchanged.
# --------------------------------------------------------------------------
AuthorityDuringSearch = Loop(
    lambda candidate_idx, exhausted: not exhausted,
    module(base_level, modifiers, required_level, name="authority"),
    carries=["authority_level_code", "authority_sequence", "self_approval_flag"],
    max_iterations=2_200,     # the candidate space bound (05 5.12)
)


def self_approval(structurer_id: int, approver_level: int,
                  second_signature_floor_cents: int = param(1_00_000_00, ge=0)) -> bool:
    """Spec 5.14.3 requirement 4. Sequence (c) is the reason.

    Detected, not prevented -- because preventing it would mean forbidding an
    analyst from improving a structure, which is the behaviour the Bank wants.
    Above R1 000 000 a second signature is required.
    """
    pass  # compare structurer against the level that would approve


ApprovalBinding = approval_binds_to("proposed_structure_fingerprint")
"""Spec 5.14.3 requirement 3, and the reason this is a framework construct
rather than a rule.

    "A structure change after approval that would have required a higher
     authority INVALIDATES THE APPROVAL. Explicitly, automatically, and with the
     invalidation recorded."

An approval is bound to a content hash of the structure it approved. Any
downstream write to a name inside that fingerprint's lineage invalidates it and
emits reason 5894. Doing this as a rule means somebody has to remember to write
the rule for every one of the 1 900 decision points that can move a structure;
doing it as a binding over the lineage the framework already computes means the
14 teams cannot collectively forget. FRAMEWORK-DEMANDS D11.
"""


def pack_sections(authority_level_code: int) -> list:
    """322-cell matrix, 7 levels x 46 sections. Spec 5.14.2.

    The pack is GENERATED, never assembled by hand (05 9.3), and requirement 5
    is that it REGENERATES WHEN THE LEVEL CHANGES: "a level 6 pack describing a
    structure that was subsequently reduced to a level 4 decision is a
    governance failure that looks like a formatting problem."

    So the pack is derived from `ApprovalBinding`'s fingerprint, not from a
    variable somebody sets. If the fingerprint moved, the pack is stale, and a
    stale pack cannot be presented -- the same mechanism, reused.
    """
    pass  # section list from the matrix; render from the decision record


Authority = module(base_level, modifiers, required_level, self_approval,
                   pack_sections, name="authority",
                   owner="credit_governance", approved_by="board_credit_committee",
                   taps=["authority_level_code@*", "self_approval_flag", "branch_path"])
