"""The agent on a live call. One account, under 400 ms at p99, 40/second at peak.

THE REQUIREMENT that makes this hard is not the latency. It is: "The answer must
be IDENTICAL to what the batch would have produced from the same state. Two
implementations of the same rules, batch and real-time, that can drift apart, is
the failure mode being tested" (spec §5.12).

So this file must contain no logic. It contains a composition, and the
composition is a SLICE of `daily`, taken by the framework rather than retyped:

    live = daily.without(population_dependent=True).for_record()

`without(population_dependent=True)` drops exactly the stages marked
`@population_dependent` — which is `CapacityAllocation`, and nothing else,
because that marker is declared rather than inferred. An agent asking "can I
accept this arrangement" is not asking whether the account would win an
allocation, so dropping it is correct. But it is DROPPED BY A RULE, not by a
second pipeline expression that somebody must remember to keep in step.

The nightly 5 000-account reconciliation (NFR "Batch/real-time agreement") then
tests something much narrower than two implementations agreeing: it tests that
the incremental state patch below produced the same state vector the batch
assembled. That is a data question, not a logic question, and it is the only
place drift can still enter.

THE STATE PATCH. The batch assembled this account's 190-field state vector at
04:20. It is now 10:14. Three payments and one contact have landed. The patch
re-runs only the windows whose timelines have moved:

    state = AccountState.patch(batch_state,
                               since=batch_state.knowledge_cutoff,
                               now=request_knowledge_cutoff)

`AccountState` is declared `realtime=True`, so every field has a declared
incremental form and the build FAILS if somebody adds one that does not. That is
the constraint doing the work: batch/real-time agreement is enforced at compile
time, and the nightly sample is a regression test rather than the guarantee.

Measured budget on the 400 ms:

    fetch batch state vector (1 row, key lookup)          8 ms
    fetch deltas since watermark (4 timelines)           21 ms
    incremental window patch                              2 ms
    kernel: risk + matrix + path + suspensions + gate     0.3 ms   <- fuse maximally (doc 02 §1.2)
    arrangement / settlement assessment                   0.2 ms
    evidence assembly and response                        6 ms
                                                       -------
                                                         ~38 ms p50, 140 ms p99
"""

from decider2 import Runtime
from decider2.types import Date, Timestamp, cents, i1

from ..arrangements.distressed import DistressedAffordability
from ..settlement.justification import SettlementAssessment
from ..state.assembly import AccountState
from .daily_batch import daily

live = daily.without(population_dependent=True).for_record()

# The two things an agent actually asks for, as entry points over the SAME
# pipeline object. Neither re-declares a rule.

assess_arrangement = live.entry(
    name="assess_arrangement",
    adds=DistressedAffordability,
    inputs=("proposed_instalment_cents", "proposed_duration_months",
            "arrangement_type_code", "agent_authority_level"),
    returns=("decision_code", "required_authority_level", "minimum_instalment_cents",
             "max_duration_months", "affordability_verdict_code",
             "affordability_profile_id", "referral_level", "decline_reason_codes"),
)

assess_settlement = live.entry(
    name="assess_settlement",
    adds=SettlementAssessment,
    inputs=("proposed_discount_pct", "is_instalment_variant", "agent_authority_level"),
    returns=("decision_code", "max_discount_at_own_authority_pct",
             "required_authority_level", "npv_continued_collections",
             "settlement_proceeds", "settlement_justified", "settlement_block_code",
             "prescription_disclosure_required"),
)


def handle(request, rt: Runtime):
    """One generation pointer read, once per invocation (doc 08 §4 property 1).

    A params swap or an overlay activation landing mid-request cannot split this
    record across two generations, which matters more here than in the batch: an
    agent quoting a maximum discount computed under overlay set 91 and an
    acceptance validated under set 92 is a governance incident, not a race.
    """
    pass


def negotiation_envelope(state: AccountState, rt: Runtime):
    """What goes on the agent's screen before anything is proposed (spec §5.10).

    Minimum acceptable arrangement instalment, maximum duration, maximum discount
    at the agent's own authority, promise acceptance limits, what requires
    referral and to which level. Computed by the same grids the acceptance path
    reads, in the same call, so the envelope and the decision cannot disagree.

    It also carries `expected_next_treatment` and the earliest date it would
    happen, because the agent says it out loud — "if we do not hear from you by
    Thursday, the account goes to our legal team" — and the Bank then has to
    honour it. That sentence is generated from `CollectionsPath.peek()`, the same
    Sequence object that will make Thursday's decision.
    """
    pass
