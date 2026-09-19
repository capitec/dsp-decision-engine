"""The consolidation flow. Composition only - no logic lives in this file.

Eleven stages from the spec, and the whole flow fits on one screen because every
stage is a module and the ordering is visible in the `|`.

    5.1  intake and eligibility
    5.2  obligation inventory and settleability
    5.3  settlement amount derivation
    5.4  baseline assessment and the short-circuit
    5.5  candidate scenario generation      \\
    5.6  per-scenario evaluation             > Search(...)
    5.7  policy interventions                /
    5.8  objective and selection            /
    5.10 output and execution package

THE ONE THING THIS FILE ASSERTS THAT DOC 03 CANNOT EXPRESS:

`Search` is a FOURTH COMBINATOR. Doc 03 8 has three - `|` (sequence), `Branch`
(conditional) and `Loop` (bounded iteration) - and says "there is one type,
Module, and three combinators over it", which is a design claim I am departing
from deliberately.

A bounded search is not a Loop. A Loop carries accumulators through a body and
asks "should I continue". Expressing this search as a Loop would mean:

  - the candidate PLAN hidden inside a step body, where the ordering rules stop
    being an artefact Credit Risk Policy can author and become code;
  - the per-candidate evaluation running one row at a time, forfeiting the
    vectorised kernel that is the only reason 400 scenarios fit in 900 ms;
  - the rejection record accumulating in a carry, which is a variable-length
    structure inside a compiled loop and therefore not expressible at all;
  - and the budget being a `max_iterations`, which conflates the deterministic
    count bound with the latency bound that must never decide the answer.

Every one of those is a real loss and the first is the worst. See FRAMEWORK-
DEMANDS D1.
"""

from decider2 import Branch, module
from decider2.frame import Join
from decider2.search import Search

from credit_core import affordability, bureau, consent, deductions, eligibility, expense_norms, income
from inventory.baseline import Baseline
from inventory.settleability import Settleability
from inventory.settlement_amount import SettlementAmounts
from output.comparison import BeforeAndAfter
from output.execution_package import ExecutionPackage
from output.explain import RejectionExplanation
from policy.interventions import PolicyInterventions
from search.budget import InteractiveBudget, SearchEvidence
from search.evaluate import EvaluateScenario
from search.plan import ConsolidationPlan
from search.select import SelectWinner, Selection


# --- 5.1 intake and eligibility -----------------------------------------------
#
# The library's gates apply unchanged. This flow adds seven of its own, and the
# ORDER MATTERS because several are cheap and terminal.
#
# CON-ELIG-01 is the one that is easy to get wrong: a client under debt review is
# ROUTED, not declined. A debt review decline reads as a credit refusal and is
# not one. It is a Branch, here, in the composition - not an early return inside
# an eligibility step, where it would be invisible in the rendered artefact and
# where a later refactor could turn it back into a decline.

Intake = (
    eligibility.Gates
    | module("con_elig_gates", name="consolidation_eligibility")
    | Branch(
        "eligibility_route",
        {
            0: module("proceed", name="proceed"),
            1: module("route_debt_counsellor", name="debt_counsellor_route"),  # CON-ELIG-01
            2: module("terminal_decline", name="terminal"),                    # CON-ELIG-02, 05, 06
            3: module("refer_compliance", name="refer"),                       # CON-ELIG-03, 07
        },
        modifies=["outcome_code", "primary_reason_code", "referral_queue_code"],
        taps=["branch_path"],
    )
)


# --- 5.2-5.4 the inventory, the amounts, the baseline -------------------------
#
# Income is established HERE, once, on a one-row frame. Nothing downstream can
# re-derive it because nothing downstream imports it.

Assessment = (
    bureau.Normalise
    | Settleability
    | SettlementAmounts
    | income.Determine
    | deductions.Statutory
    | expense_norms.Apply
    | Baseline
)


# --- 5.5-5.8 the search -------------------------------------------------------
#
# The whole centre of the project, as one declaration.

ScenarioSearch = Search(
    name="consolidation_search",
    # 5.5 - frame tier. Produces the ordered, deduped, truncated, tiered plan.
    plan=ConsolidationPlan,
    # 5.6 - record tier. Applied to the plan as a FRAME, once per tier.
    evaluate=EvaluateScenario,
    # 5.7 - record tier, same frame, fourteen verdict columns out.
    admit=PolicyInterventions,
    # 5.8 - frame tier reduction. Declared, so lineage survives it.
    select=SelectWinner,
    # 5.5.1 - two bounds doing two jobs.
    budget=InteractiveBudget,
    # 5.6.1 - PROVABLY constant across every scenario. Checked at runtime with one
    # n_unique() per column, which is the cheapest acceptance criterion in the
    # spec (AC 8) and the only one that would otherwise be a matter of discipline.
    invariant=[
        "gross_monthly_income",
        "income_source_code",
        "income_haircut_applied",
        "statutory_deductions",
        "net_monthly_income",
        "living_expenses",
        "expense_basis_code",
        "dependants_count",
        "baseline_total_commitment",
        "baseline_total_remaining_cost",
        "baseline_weighted_rate",
    ],
    # 5.5.5 and 5.5.6 - what a replay has to pin.
    records=[
        "plan",             # the ordered candidate list, with generation order
        "verdicts",         # fourteen columns x 400 rows, PASS/FAIL/NA
        "actuals",          # actual and threshold per failing intervention
        "measures",         # every scenario's objective score, decomposed
        "budget",           # declared, consumed, termination cause
        "overlay_stack",    # members, scopes, composition order, base values
        "table_versions",   # as a SET, so incompatible combinations are detectable
    ],
    evidence=SearchEvidence,
)


# --- 5.10 output ---------------------------------------------------------------

Output = BeforeAndAfter | ExecutionPackage | RejectionExplanation | Selection


# --- the pipeline ---------------------------------------------------------------

pipeline = Intake | Assessment | ScenarioSearch | Output


# --- what a replay pins ---------------------------------------------------------
#
#   the skeleton identity          framework + module distribution versions
#   the structure fingerprint      pipeline.fingerprint(), skeleton AND interiors
#   the compiled artefact id       generated source + signatures + variants + CPU
#   the params digest              canonical resolved values
#   the params origin              the caller's token, verbatim
#   THE OVERLAY STACK              members, scopes, order, effects, base values
#   THE PLAN DIGEST                a content hash of the ordered candidate list
#   the table version SET          all of them, together, so cross-table
#                                  inconsistency is detectable rather than merely
#                                  present
#
# The last three are this project's additions to doc 08 8's list, and each closes
# a hole doc 08 8 leaves open for a search:
#
#   without the overlay stack, AC 3 fails - a replay under a different stack
#   produces a different winner and it reads as a defect rather than an
#   explanation;
#
#   without the plan digest, "was the search the same?" needs the plan re-run,
#   which needs every input that produced it, which is the whole assessment;
#
#   without the version SET, an assessment reading September's vehicle guide
#   against August's Drive Finance rate card is not wrong so much as
#   unattributable, and spec 6.1 asks for exactly this and the library does not
#   provide it.
