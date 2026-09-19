"""Stage 5.5 -- the cap waterfall.  52 rules, three ceilings, four owners.

WHY THIS IS NOT `SeedTermCap | ApplyIncomeCap | ApplySectorCap`.

Doc 03 §3.2 models a waterfall as a chain of `term_cap -> term_cap` modules
composed with `|`, and gets attribution from the value-version chain.  That is
right for three rules and wrong for this register, for three independent
reasons:

1.  **Order is config here, and `|` is code.**  The register is reordered and
    added to QUARTERLY by four different teams, and acceptance criterion 15
    says adding a rule in any sequence position is "a configuration change
    reviewed by its owner -- not a release".  Sequence position in a `|`
    expression is a Python edit, a build, a compile and a deploy.

2.  **The version chain records changes; the spec requires verdicts.**  §5.5
    demands, for all 52 rules, the distinction between "did not apply" and
    "applied and did not bind" -- and a rule that applies and does not bind
    produces NO version, so the version chain cannot represent it.  Client W's
    chain has four bound rows and six non-bound rows, and a policy owner asking
    "is my rule doing anything?" is asking about the six.

3.  **Fifty-two modules is fifty-two kernels.**  Doc 02 §1.1 measured fusion at
    0.22x on 40 modules and split kernels at a flat 0.20 ns/step, so the
    performance is survivable either way -- but the audit artefact is 52
    entries in a pipeline expression that four teams edit concurrently, which
    is a merge-conflict generator, not a governance boundary.

So: a `Waterfall` kind.  The register is tabular -- N rules x fixed attributes
with a closed effect vocabulary -- which by doc 08 §3.4's own test puts it on a
GENERIC KERNEL.  One compiled loop over the rule arrays.  Adding a rule at
sequence 27 is a row in a JSON document: no codegen, no compile, no stage, no
release.  Criterion 15 is then true by construction rather than by process.

See FRAMEWORK-DEMANDS #4 and #5.
"""

from __future__ import annotations

from decider2 import Waterfall
from decider2.money import Money
from decider2.waterfall import Ceiling, Direction, Effect, TieBreak

from modules.caps import predicates, uplift

CapWaterfall = Waterfall(
    name="cap_register",
    # ------------------------------------------------------------------
    # The three ceilings.  `direction` is declared HERE, in code, under a
    # release -- which is what makes §13.19's "impossible to bypass by
    # defining a negative magnitude" true.  A register row whose effect would
    # move a ceiling the wrong way is rejected at interior validation, naming
    # the ceiling and its declared direction.  The register cannot grant
    # itself the right to raise, because the right is not in the register.
    # ------------------------------------------------------------------
    ceilings={
        "amount_cap": Ceiling(seed=Money("500000.00"), direction=Direction.REDUCE_ONLY),
        "term_cap": Ceiling(seed=84, direction=Direction.REDUCE_ONLY),
        "worst_acceptable_grade": Ceiling(seed=12, direction=Direction.TIGHTEN_ONLY),
    },
    # The ONE exception, named in code.  CAP-0420 may raise `amount_cap` by up
    # to `uplift_maximum`, subject to `uplift_authority_ceiling`, and subject to
    # never exceeding a ceiling set by any rule of class `regulatory`.  Its
    # authority reference is recorded whenever it fires; when it is RESTRAINED
    # -- authorised to reach a value it was not permitted to reach -- both the
    # authorised value and the restraining rule are recorded.
    raise_permitted_by={"CAP-0420": uplift.CAMPAIGN_UPLIFT_AUTHORITY},
    # ------------------------------------------------------------------
    # The closed effect vocabulary.  Six verbs.  A register row may use these
    # and nothing else; there is no expression string, so codegen is total and
    # an interior that validates always compiles (doc 08 §3 property 2).
    # ------------------------------------------------------------------
    effects=[
        Effect.REDUCE_TO,       # ceiling = min(ceiling, value)
        Effect.SCALE_BY,        # ceiling = ceiling * ratio      (order-sensitive)
        Effect.REDUCE_BY,       # ceiling = ceiling - absolute
        Effect.TIGHTEN_TO,      # grade   = min(grade, value)
        Effect.DECLINE,         # outright, with a reason code
        Effect.RAISE_TO_BOUNDED,  # CAP-0420 only; bounded by authority + regulatory class
    ],
    # Applicability predicates are REGISTERED STEPS referenced by id, never
    # expressions in config (doc 08 §3.2).  `reads` is therefore an upper
    # bound: the transitive closure of every referenced predicate's inputs must
    # fit inside it, checked at interior validation before any compile.
    predicates=predicates.REGISTRY,
    reads=[
        "risk_grade", "segment_code", "channel_code", "campaign_id",
        "months_employed", "worst_arrears_months", "months_since_worst_arrears",
        "enquiry_velocity_60d", "employer_id", "employer_on_watchlist",
        "group_exposure_limit", "internal_exposure_total", "product_code",
        "is_joint_application", "internal_tenure_months",
    ],
    capacity=64,            # register has run 44-58; sized 40-60; declared, like all capacities
    # ------------------------------------------------------------------
    # The attribution requirement.  Four outputs, not one.
    # ------------------------------------------------------------------
    tie_break=TieBreak.EARLIEST_IN_SEQUENCE,   # two rules reaching the same value:
                                               # the earlier BINDS, the later is COINCIDENT
    writes=[
        # 1. the final value
        "amount_cap", "term_cap", "worst_acceptable_grade",
        # 2. which rule set it -- exactly one answer, unambiguous
        "amount_cap_bound_by", "term_cap_bound_by", "worst_acceptable_grade_bound_by",
        # 3. the full chain: every successive value with its producer, in order
        "amount_cap_chain", "term_cap_chain", "worst_acceptable_grade_chain",
        # 4. per-rule verdicts, all 52
        "cap_rule_verdicts",
        # and the outright declines
        "cap_decline_reason_codes",
    ],
    # `Verdict` is five-valued, not boolean.  A rule that did not apply and a
    # rule that applied but did not bind are different FACTS, and the fifth
    # value exists because a coincident reducer is neither of those either.
    #   NOT_APPLICABLE (with the predicate that failed) | EVALUATED_NOT_BINDING
    #   | BOUND | COINCIDENT | DECLINED
    verdict_kind="five_valued",
    chain_kind="ragged",    # capacity = capacity of the register + 1 seed + overlay rows
    interior="config/flex_loan/interiors/cap_register.json",
    # The generated review artefact.  This -- not the Python -- is what a
    # Credit Risk Policy analyst signs off (acceptance criterion 17).  It is
    # generated FROM the executing interior, so it cannot drift from it.
    review_artefact="review/cap_register.md",
)
