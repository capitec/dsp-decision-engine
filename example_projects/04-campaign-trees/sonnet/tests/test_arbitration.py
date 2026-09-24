"""arbitration.py: Stage 6 (spec 04 §5.6) -- determinism, capacity, a reason for every
non-contact."""
from __future__ import annotations

import random

from campaign_trees import vocab
from campaign_trees.arbitration import Qualification, arbitrate


def _synthetic_qualifications(n_clients: int, campaigns_per_client: int, seed: int) -> list[Qualification]:
    rng = random.Random(seed)
    quals = []
    for i in range(n_clients):
        client_id = f"C-{i}"
        for campaign_id in rng.sample(range(1, 61), k=campaigns_per_client):
            quals.append(Qualification(
                client_id=client_id, campaign_id=campaign_id,
                priority_weight=rng.uniform(0.1, 0.95), expected_value=rng.uniform(1.0, 500.0),
                channels=(rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3, 4])),
                is_control=rng.random() < 0.05,
            ))
    return quals


def test_every_qualifying_pair_gets_a_result():
    quals = _synthetic_qualifications(500, 4, seed=1)
    results, _ = arbitrate(quals, channel_capacity={1: 100, 2: 400, 3: 50, 4: 600})
    assert len(results) == len(quals)


def test_channel_capacity_is_never_exceeded():
    quals = _synthetic_qualifications(2_000, 5, seed=2)
    capacity = {1: 300, 2: 1_000, 3: 150, 4: 2_000}
    results, reports = arbitrate(quals, channel_capacity=capacity)
    used_by_channel: dict[int, int] = {}
    for r in results:
        if r.contacted:
            used_by_channel[r.channel] = used_by_channel.get(r.channel, 0) + 1
    for ch, cap in capacity.items():
        assert used_by_channel.get(ch, 0) <= cap
    for report in reports:
        assert report.used <= report.allocated
        assert report.used + report.unused == report.allocated


def test_every_non_contact_carries_a_reason():
    """§5.6 requirement 3: "'Not selected' without a reason is not acceptable output.\""""
    quals = _synthetic_qualifications(1_000, 6, seed=3)
    results, _ = arbitrate(quals, channel_capacity={1: 50, 2: 100, 3: 20, 4: 150})
    for r in results:
        assert r.contacted or r.reason is not None


def test_control_clients_are_never_contacted():
    quals = _synthetic_qualifications(500, 3, seed=4)
    results, _ = arbitrate(quals, channel_capacity={1: 10_000, 2: 10_000, 3: 10_000, 4: 10_000})
    for q, r in zip(sorted(quals, key=lambda q: (q.client_id, q.campaign_id)),
                     sorted(results, key=lambda r: (r.client_id, r.campaign_id))):
        if q.is_control:
            assert not r.contacted and r.reason == vocab.IS_CONTROL_GROUP


def test_cycle_contact_cap_is_respected_per_client():
    quals = _synthetic_qualifications(300, 9, seed=5)  # 9 qualifications per client, cap is 2
    results, _ = arbitrate(quals, channel_capacity={1: 100_000, 2: 100_000, 3: 100_000, 4: 100_000},
                            cycle_contact_cap=2)
    contacts_per_client: dict[str, int] = {}
    for r in results:
        if r.contacted:
            contacts_per_client[r.client_id] = contacts_per_client.get(r.client_id, 0) + 1
    assert all(n <= 2 for n in contacts_per_client.values())


def test_arbitration_is_deterministic_and_re_runnable():
    """§5.6 requirements 1-2."""
    quals = _synthetic_qualifications(800, 5, seed=6)
    results_a, _ = arbitrate(quals, channel_capacity={1: 200, 2: 500, 3: 80, 4: 700})
    results_b, _ = arbitrate(quals, channel_capacity={1: 200, 2: 500, 3: 80, 4: 700})
    assert results_a == results_b
    # order independence: shuffling the input must not change the outcome
    shuffled = list(quals)
    random.Random(999).shuffle(shuffled)
    results_c, _ = arbitrate(shuffled, channel_capacity={1: 200, 2: 500, 3: 80, 4: 700})
    key = lambda r: (r.client_id, r.campaign_id)
    assert sorted(results_a, key=key) == sorted(results_c, key=key)
