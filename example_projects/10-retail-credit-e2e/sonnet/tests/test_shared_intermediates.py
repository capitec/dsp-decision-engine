"""Shared intermediates registry (spec 10 §5.21), including the two-live-version case
(§5.21.1), proven for real in test_consolidation.py -- this module checks the declared
registry itself.
"""
from retail_credit.shared_intermediates import REGISTRY, consumers_of, two_version_concepts


def test_decision_date_is_read_by_every_phase():
    assert len(consumers_of("decision_date")) == 17  # every phase except P01, which produces it


def test_existing_obligations_has_two_live_versions():
    assert "existing_obligations" in two_version_concepts()


def test_every_registry_entry_has_at_least_two_consumers():
    for si in REGISTRY:
        assert len(si.consumed_in) >= 2, si.name
