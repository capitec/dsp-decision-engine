"""CAP-0420 -- the one rule permitted to RAISE a cap.

This is fifteen lines of code rather than a register row because the register's
direction declaration forbids raising, and the exception must be visible in
code, under a release, with a named authority -- not expressible by any Credit
Risk Policy owner who happens to be editing the register that quarter.

The restraint recording is the part that is easy to get wrong.  §5.5: "if the
uplift is *restrained* by a ceiling -- that is, if it was authorised to raise
to a value it was not permitted to reach -- both the authorised value and the
restraining rule must be recorded."  So the rule emits three values, not one:
what it was authorised to reach, what it actually reached, and (when those
differ) which rule restrained it.
"""

from __future__ import annotations

from decider2 import param, raise_authority
from decider2.money import Money

CAMPAIGN_UPLIFT_AUTHORITY = raise_authority(
    rule_id="CAP-0420",
    ceiling="amount_cap",
    # Bounded three ways.  All three are params, all three Credit-Committee-owned,
    # all three quarterly.  None of them is in the register document.
    max_ratio=param(1.25, ge=1.0, le=1.5, owner="credit_committee"),
    absolute_ceiling=param(Money("250000.00"), owner="credit_committee"),
    # The fourth bound is structural: never above a ceiling set by any rule of
    # class `regulatory`, whatever the register's order.  `not_above_class` is
    # evaluated against the chain, so it holds regardless of where in the
    # sequence the uplift sits -- which matters, because the register is
    # reordered quarterly by people who do not know this constraint exists.
    not_above_class="regulatory",
    requires=["campaign_authorised"],
    # May not touch the other two ceilings, at all, ever.
    may_not_affect=["term_cap", "worst_acceptable_grade"],
    records=["authority_reference", "authorised_value", "achieved_value",
             "restrained_by_rule_id"],
)
