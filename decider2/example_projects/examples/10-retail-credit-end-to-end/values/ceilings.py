"""The five narrowed ceilings, and the declaration that makes N1 generatable.

Spec 5.10: P09 narrows five quantities from their product seeds, and afterwards
four things must be answerable for each -- the final value, WHICH ENTRY SET IT
(exactly one answer, unambiguous), the full chain with every successive value in
order, and which entries were evaluated-and-did-not-bind versus not-applicable,
"and they are different facts".

Spec 5.27 N1: for `amount_cap` there are 132 places it can be set -- 118 register
entries (47 applicable to product 10), 6 cap overlays, 6 product maxima, 1
regulatory maximum and 1 uplift entry -- and "a human cannot enumerate 132 places
correctly by reading, and will not try twice.  The enumeration must be GENERATED,
and the requirement is therefore on the structure, not on whoever writes the
documentation."

---------------------------------------------------------------------------
THE MOVE
---------------------------------------------------------------------------
A ceiling is not a value that happens to get overwritten.  It is a declared
MONOTONE ACCUMULATOR with a seed, a direction, a chain and an attribution
contract.  Doc 03 3.2 gives the mechanism for the overwrite --
`SeedTermCap | ApplyIncomeCap | ApplySectorCap`, each rule its own auditable
unit, a version chain per boundary crossing -- and stops there.  It has no
notion of direction, no notion of "evaluated and did not bind" (a step that
returns its input unchanged is indistinguishable from one that did not run),
and no way to say that exactly one entry may go the other way.

So:

    ceiling("amount_cap", direction=REDUCE_ONLY, seed=..., may_raise=["CAP-0118"])

buys four things that are otherwise tests, comments or hope:

  1. DIRECTION IS CHECKED AT RUNTIME.  A register entry that raises `amount_cap`
     and is not CAP-0118 is a hard failure, per decision, not a quarterly
     discovery.  Spec 5.10: CAP-0118 "is the only entry permitted to raise a
     ceiling", and the authority ceiling and the never-above-regulatory rule sit
     on the ceiling, not inside the entry, so a new entry cannot acquire them.

  2. `evaluated_did_not_bind` BECOMES A RECORDED FACT.  The runtime knows the
     entry was applicable, ran, and returned the same value.  Spec 5.10
     requirement 4 and spec 5.29.1: without it, "why did CAP-0212 not bind" is
     unanswerable and the dead-logic measurement collapses into firing counts.

  3. COINCIDENCE IS RESOLVED BY RULE, NOT BY ORDER OF ARGUMENTS.  "Where two
     entries reduced to the same value, the earlier is the binder and the later
     is recorded as coincident" -- spec 5.10 requirement 2.  That is one line
     here instead of an unwritten convention in 118 entries.

  4. N1 IS GENERATABLE.  Every `narrows=` in the repo is an edge into this
     declaration, so `where("amount_cap")` enumerates 132 places because the
     structure enumerates them, not because someone maintained a list.
"""

from __future__ import annotations

from decider2.values import REDUCE_ONLY, TIGHTEN_ONLY, ceiling

amount_cap = ceiling(
    "amount_cap",
    dtype="Money",
    direction=REDUCE_ONLY,
    seed="product_maximum",                 # per candidate product
    may_raise=["CAP-0118"],                 # the ONLY entry.  Spec 5.10.
    raise_bounded_by={"relative": 0.20, "absolute": 2_000_00, "never_above_class": "regulatory"},
    coincidence="earlier_binds",
    chain_recorded=True,
    overlay_rows_attributed_to="adjustment_set",   # NOT to a register entry
    # Spec 5.10: "your cap was R134 400, reduced to R120 750 by a policy overlay
    # approved under CC-2027-11, expiring 2027-12-31" is a different answer to a
    # client and to a regulator than "a rule bound it".  A cap overlay enters the
    # chain as its OWN row and may only reduce, so it can never do what
    # CAP-0118's authority-bounded uplift does.
    rounding_grid=250_00,
    rounding_applied_at="chain_end_only",    # O-22: exactly one place per value
)

term_cap = ceiling(
    "term_cap", dtype="int16", direction=REDUCE_ONLY, seed="product_maximum",
    may_raise=[], coincidence="earlier_binds", chain_recorded=True,
)

limit_cap = ceiling(
    "limit_cap", dtype="Money", direction=REDUCE_ONLY, seed="product_maximum",
    applies_to_products=(20, 21), may_raise=[], chain_recorded=True,
)

worst_acceptable_grade = ceiling(
    "worst_acceptable_grade", dtype="int8", direction=TIGHTEN_ONLY, seed=12,
    may_raise=[], chain_recorded=True,
    # L7, spec 5.23: CAP-0176 tightens this, and the tightened grade re-keys the
    # appetite grid CAP-0104 read EARLIER in the same register.  The declared
    # answer is that the register is single-pass and earlier entries do not
    # re-run -- which means register order is semantically load-bearing.  That is
    # an arguable trade, so it is declared here rather than discovered:
    reentrant=False,
    order_is_load_bearing=True,
    surfaces_in_reviewable_artefact=True,
    note="L7. A quarterly reordering can change outcomes for reasons nobody "
         "intended. Rendered at the top of the P09 reviewable section, not "
         "buried, because a policy owner will not discover it by reading their "
         "own rule.",
)

instalment_cap = ceiling(
    "instalment_cap", dtype="Money", direction=REDUCE_ONLY,
    seed="p10.max_affordable_instalment",
    # O-10, the second genuine cycle.  Affordability produces the seed and two
    # register entries reduce it further, so the register runs in two passes and
    # THE ENTRIES DO NOT KNOW WHICH PASS THEY ARE IN.  The pass is derived from
    # which ceiling an entry declares `narrows=`, not declared per entry -- which
    # is what keeps the split invisible in the register, as the spec requires.
    pass_derived_from="narrows",
    chain_recorded=True,
)

CEILINGS = (amount_cap, term_cap, limit_cap, worst_acceptable_grade, instalment_cap)

# Asserted by `decider2 where amount_cap --count`.  Spec 5.27 N1.
EXPECTED_AMOUNT_CAP_SET_SITES = 132
