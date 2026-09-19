"""The restructure option set: forbearance the Bank GRANTS, not credit it advances.

Spec 5.9. "The machinery of 5.5 to 5.8 is reused unchanged. What changes:" -
the objective, the option set, the affordability test, and who may approve.

That sentence is the acceptance criterion for the whole design. If the
restructure variant needs its own search, its own selection, its own intervention
mechanism or its own record shape, the machinery was not general; it was one
flow with a search-shaped hole in it.

What actually changes is ONE STAGE: what a candidate is. A consolidation
candidate is a settlement subset carried by a product over a term. A restructure
candidate is a COMBINATION OF CONCESSIONS over existing agreements. Both are rows
in a plan frame with a candidate key; everything after generation is identical.

The combination is what enlarges the space again. A payment holiday followed by a
term extension is a distinct option from either alone, so the option set
multiplies the settlement-set space rather than replacing it - and that is why
the restructure budget declares 260 candidates rather than 400 despite having
fewer accounts in play. The per-candidate evaluation is more expensive (two
affordability verdicts, an NPV, an authority determination, a distressed
classification), not less numerous.
"""

from decider2 import module, param, step
from decider2.search import Plan
from decider2.values import Maybe, na


# --- the eight concessions, as a catalogue -----------------------------------
#
# 8 concessions x 4 authority levels x 5 attributes = 160 cells, owned by the
# Credit Committee, changed annually. A table, not code - the BOUNDS are values
# and the arithmetic of each concession's effect is a step.


@step(output="concession_effect")
def concession_effect(
    concession_code: int,
    balance: float,
    instalment: float,
    nominal_annual_rate: float,
    remaining_term_months: int,
    magnitude: float,
) -> dict:
    """Apply one concession to one agreement, returning the post-concession terms.

    CNC-01 payment holiday          1-3 months, interest CONTINUING TO ACCRUE and
                                    capitalising. The capitalisation is the part
                                    clients do not expect and the part the
                                    comparison must show.
    CNC-02 term extension           up to +36 months on an existing internal
                                    agreement, not beyond the product maximum.
    CNC-03 rate concession          up to -400bp, for 6, 12 or 24 months, or for
                                    the remaining term.
    CNC-04 arrears capitalisation   up to 6 months of arrears folded into the
                                    balance.
    CNC-05 temporary reduction      50-80% of contractual for 6 or 12 months,
           with step-up             then contractual or above. The step-up is
                                    contractual and must be in the comparison.
    CNC-06 fee and interest waiver  up to R15 000.
    CNC-07 partial capital          highest authority only, always.
           forgiveness
    CNC-08 consolidation into       where affordability supports it - which is
           product 11 at a          the point where the restructure variant
           concessionary rate       re-enters the consolidation machinery, using
                                    the same product arm with a concessionary
                                    rate overlay rather than a second product.
    """
    pass  # per-concession arithmetic over the agreement's terms


# --- the plan: combinations, bounded -----------------------------------------
#
# The same Plan vocabulary as search/plan.py. Seeds, prefixes, distinct, truncate,
# tier - identical operations over a different element type, which is the
# evidence that the plan machinery is general rather than consolidation-shaped.

RestructurePlan = (
    Plan.seed("do_nothing", concessions=[])
    | Plan.seed("single", concessions="each_permitted_concession")
    # Combinations to depth 2, ordered by NPV cost ascending - cheapest first,
    # because an authority level is a cost and the cheapest sustainable option is
    # the one that does not need escalating.
    | Plan.combinations(of="permitted_concessions", depth=param(2, ge=1, le=3))
    | Plan.order_by("npv_cost asc", "sustainability_estimate desc")
    | Plan.distinct(on="concession_combination_key")
    | Plan.truncate(to="budget.candidates")
    | Plan.tier(by="budget.tiers")
    | Plan.cross(source="assessment_invariants", broadcast=True)
)


# --- NPV cost and the authority it requires ----------------------------------


@step(output="npv_cost")
def npv_cost(
    contractual_cashflows: list,
    concession_cashflows: list,
    discount_rate: float,
) -> float:
    """Net present value forgone against the contractual position.

    The number that decides who may approve, so it is computed before approval is
    sought rather than estimated afterwards, and it is recorded per OPTION - not
    only for the one granted. Internal Audit's sample of forty files asks for the
    NPV cost of the concession granted; a file that cannot also show what the
    alternatives cost cannot show that the cheapest sustainable option was
    chosen.
    """
    pass  # PV(contractual) - PV(concession), at the declared discount rate


@step(output="authority_level_code")
def authority_level(
    npv_cost: float,
    concession_code: int,
    prior_concession_months_ago: int,
    band_1_max: float = param(5000.0, ge=0.0, le=20000.0),
    band_2_max: float = param(25000.0, ge=5000.0, le=100000.0),
    band_3_max: float = param(150000.0, ge=25000.0, le=500000.0),
) -> int:
    """1 consultant / 2 team leader / 3 Credit Risk manager / 4 Credit Committee.

    Above R150 000, and ANY CNC-07 (partial capital forgiveness), is level 4
    regardless of amount. A SECOND CONCESSION TO THE SAME CLIENT WITHIN 12 MONTHS
    RAISES THE REQUIRED AUTHORITY BY ONE LEVEL - which means the same option
    carries a different authority for two clients with the same numbers, and the
    reason is in the record.
    """
    pass  # band lookup, then CNC-07 floor, then the repeat-concession bump


@step(output="authority_outcome")
def authority_outcome(
    authority_level_code: int,
    running_actor_authority: int,
) -> int:
    """APPLY where the running actor holds the authority; REFER otherwise.

    A referral carries everything the approver needs: the option, its NPV cost,
    both affordability verdicts, the distressed classification, the alternatives
    considered and why this one, and the client's history of prior concessions.

    Internal Audit's finding, stated in spec 9.2: "a file where the approving
    authority is recorded as THE SYSTEM is a finding." So the record carries the
    IDENTITY of the approver, and where the system applied a level-1 concession
    the record carries the consultant's identity and their held authority - not
    "auto-approved".
    """
    pass  # APPLY | REFER, with the referral package assembled


# --- distressed or commercial -------------------------------------------------


@step(output="distressed_classification")
def distressed_classification(
    concession_code: int,
    months_in_arrears: int,
    hardship_declared: bool,
    npv_cost: float,
    balance: float,
    prior_concession_months_ago: int,
) -> int:
    """DISTRESSED or COMMERCIAL, with the basis recorded, BEFORE the concession is granted.

    A concession granted because of financial distress may constitute a
    distressed restructure, with consequences for provisioning classification and
    for the client's bureau status.

    "Before, not after, when the provisioning report finds it" is the requirement,
    and it is a sequencing requirement rather than a computational one: the
    classification is cheap and the flow could compute it anywhere. Putting it in
    the per-candidate evaluation means the consultant sees it while discussing
    the option, which is the only point at which it can change anything.
    """
    pass  # classification + the ordered list of facts that produced it


# --- stressed affordability, as a second instance of the same module ----------
#
# search/evaluate.py declares AffordScenarioStressed with `.at()` rebinding three
# inputs. Nothing about core.affordability is forked, wrapped or copied.
#
# Income reduced 10%, living expenses increased 8%, retained variable-rate
# obligations repriced 200bp higher. BOTH VERDICTS ARE RECORDED. An option
# passing standard and failing stressed may be offered only with a recorded
# acknowledgement that it was, and only up to authority level 2.
#
# That last clause is the interesting one: it makes the stressed verdict an input
# to the AUTHORITY determination, not only to viability. A restructure that only
# works if nothing else goes wrong is not a restructure, and the design's answer
# is not to forbid it but to price it in approval authority.


@step(output="stress_acknowledgement_required")
def stress_acknowledgement_required(
    affordability_verdict_standard: int,
    affordability_verdict_stressed: int,
    authority_level_code: int,
) -> bool:
    """Passing standard and failing stressed: permitted, acknowledged, capped at level 2."""
    pass  # True where standard passes and stressed fails


Concessions = module(
    concession_effect,
    npv_cost,
    authority_level,
    authority_outcome,
    distressed_classification,
    stress_acknowledgement_required,
    name="concessions",
    params="config/restructure/concessions.json",
    taps=["npv_cost", "authority_level_code", "distressed_classification"],
    contract="contracts/concessions.json",
)
