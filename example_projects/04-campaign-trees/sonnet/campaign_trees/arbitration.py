"""Stage 6 -- arbitration (spec 04 §5.6): which of a client's qualifying campaigns actually
result in a contact this cycle. **Deliberately not a `decider` step.** §5.6 is explicit that
this "cannot be decided one client at a time" -- channel capacity and the per-client contact
cap are population-level constraints, and a per-row `decider` step (compiled or interpreted)
has no view across rows. `pipeline.py` serves one (client, campaign) evaluation at a time,
exactly as 00 and 03's own demo pipelines serve one representative slice; this module is the
population-level pass that runs *after* a batch of Stage 3-5 evaluations exists, over plain
Python data structures, and is proven by `tests/test_arbitration.py` instead of by
`decider build`. SCOPE.md explicitly cuts "fairness reporting" (the 35% rolling-6-cycle rule
and the 60% rank-one-demand floor, §5.6's fairness rule) from this slice; what's built is
§5.6 requirements 1-4 -- determinism, re-runnability, a reason for every non-contact, and
capacity utilisation reporting.

**The mechanism**: ranking-and-cutting (one of the four the spec explicitly permits without
prescribing one, §5.6). Qualifications are sorted once, deterministically
(`priority_weight * expected_value` descending, a `(client_id, campaign_id)` tiebreak so two
equal scores never depend on input order), then walked once, allocating each client's two
cycle-cap contact slots and each channel's remaining capacity as it goes -- an O(n log n)
sort plus one O(n) pass, not a search, which is what "deterministic" and "re-runnable"
(§5.6 requirements 1-2) need in practice: no randomness, no non-deterministic dict iteration
order (`sort_key` breaks every tie), no window where two runs could allocate the same
channel slot to different clients.

**Fatigue** (the 30-day rolling cap) is enforced upstream, in `suppressions.py`'s S30 --  a
client past it never reaches Stage 3 at all, so it never appears in arbitration's own input.
A client who loses a slot *here* lost it to a higher-ranked qualifying campaign within this
one cycle's 2-contact cap, which is `LOST_ON_RANK`, not `FATIGUE_CAP_REACHED` -- the vocab
constant exists (`vocab.FATIGUE_CAP_REACHED`) for a caller that wants to attribute a
suppression-stage removal using the same reason vocabulary, not because this module produces
it itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from campaign_trees import vocab


@dataclass(frozen=True)
class Qualification:
    client_id: str
    campaign_id: int
    priority_weight: float
    expected_value: float
    channels: tuple[int, ...]   # ordered preference, most-preferred first (§5.3.2)
    is_control: bool = False
    campaign_suspended: bool = False


@dataclass(frozen=True)
class ArbitrationResult:
    client_id: str
    campaign_id: int
    contacted: bool
    channel: int | None
    reason: str | None          # None iff contacted


@dataclass(frozen=True)
class CapacityReport:
    channel: int
    allocated: int
    used: int
    unused: int
    refused_demand: int         # qualifications that wanted this channel and did not get it


def _rank_key(q: Qualification) -> tuple:
    # Descending score, then an ascending id tiebreak -- entirely a function of the
    # qualification's own identifiers, so two runs (or two orderings of the same input) sort
    # identically (§5.6 requirement 1).
    return (-(q.priority_weight * q.expected_value), q.client_id, q.campaign_id)


def arbitrate(
    qualifications: list[Qualification], channel_capacity: dict[int, int], *, cycle_contact_cap: int = 2,
) -> tuple[list[ArbitrationResult], list[CapacityReport]]:
    ranked = sorted(qualifications, key=_rank_key)
    remaining_capacity = dict(channel_capacity)
    used = {ch: 0 for ch in channel_capacity}
    refused = {ch: 0 for ch in channel_capacity}
    client_contacts: dict[str, int] = {}
    client_last_winner: dict[str, int] = {}  # client_id -> campaign_id that took their most recent slot

    results: list[ArbitrationResult] = []
    for q in ranked:
        if q.is_control:
            results.append(ArbitrationResult(q.client_id, q.campaign_id, False, None, vocab.IS_CONTROL_GROUP))
            continue
        if q.campaign_suspended:
            results.append(ArbitrationResult(q.client_id, q.campaign_id, False, None, vocab.CAMPAIGN_SUSPENDED))
            continue
        if client_contacts.get(q.client_id, 0) >= cycle_contact_cap:
            winner = client_last_winner.get(q.client_id)
            results.append(ArbitrationResult(q.client_id, q.campaign_id, False, None,
                                              f"{vocab.LOST_ON_RANK}:{winner}"))
            continue
        chosen_channel = None
        for ch in q.channels:
            if remaining_capacity.get(ch, 0) > 0:
                chosen_channel = ch
                break
            refused[ch] = refused.get(ch, 0) + 1
        if chosen_channel is None:
            results.append(ArbitrationResult(q.client_id, q.campaign_id, False, None,
                                              vocab.CHANNEL_CAPACITY_EXHAUSTED))
            continue
        remaining_capacity[chosen_channel] -= 1
        used[chosen_channel] = used.get(chosen_channel, 0) + 1
        client_contacts[q.client_id] = client_contacts.get(q.client_id, 0) + 1
        client_last_winner[q.client_id] = q.campaign_id
        results.append(ArbitrationResult(q.client_id, q.campaign_id, True, chosen_channel, None))

    capacity_reports = [
        CapacityReport(ch, channel_capacity[ch], used.get(ch, 0),
                        channel_capacity[ch] - used.get(ch, 0), refused.get(ch, 0))
        for ch in channel_capacity
    ]
    return results, capacity_reports
