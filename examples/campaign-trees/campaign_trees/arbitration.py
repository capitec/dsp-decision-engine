"""Arbitration: which of a client's qualifying campaigns become a contact this cycle.

It looks across clients (channel capacity, the per-client contact cap), so it runs over a batch of assignments, not as a
step of the one-client pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qualification:
    client_id: str
    campaign_id: int
    priority_weight: float
    expected_value: float
    channels: tuple[int, ...]  # most preferred first
    is_control: bool = False
    campaign_suspended: bool = False


@dataclass(frozen=True)
class Contact:
    client_id: str
    campaign_id: int
    channel: int | None  # None: not contacted
    reason: str | None  # why not, None when contacted


def arbitrate(qualifications: list[Qualification], channel_capacity: dict[int, int], cycle_contact_cap: int = 2) -> list[Contact]:
    """Rank by `priority_weight * expected_value` and walk down once, giving each client its best campaigns a channel.

    The order is total (ties break on client and campaign id), so the same input always gives the same contacts, and
    every client who is not contacted has a reason: `control_group`, `campaign_suspended`, `lost_on_rank:<winning
    campaign>` or `channel_capacity_exhausted`.

        arbitrate([Qualification("c1", 23, 0.8, 100.0, (2, 1))], {2: 1000, 1: 1000})  # contact c1 by sms (channel 2)
    """
    remaining = dict(channel_capacity)
    contacts: dict[str, list[int]] = {}
    results = []
    for q in sorted(qualifications, key=lambda q: (-(q.priority_weight * q.expected_value), q.client_id, q.campaign_id)):
        won = contacts.setdefault(q.client_id, [])
        channel = next((c for c in q.channels if remaining.get(c, 0) > 0), None)
        if q.is_control:
            results.append(Contact(q.client_id, q.campaign_id, None, "control_group"))
        elif q.campaign_suspended:
            results.append(Contact(q.client_id, q.campaign_id, None, "campaign_suspended"))
        elif len(won) >= cycle_contact_cap:
            results.append(Contact(q.client_id, q.campaign_id, None, f"lost_on_rank:{won[-1]}"))
        elif channel is None:
            results.append(Contact(q.client_id, q.campaign_id, None, "channel_capacity_exhausted"))
        else:
            remaining[channel] -= 1
            won.append(q.campaign_id)
            results.append(Contact(q.client_id, q.campaign_id, channel, None))
    return results
