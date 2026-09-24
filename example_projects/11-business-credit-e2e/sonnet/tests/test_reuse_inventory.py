"""The §5.17.7 report -- SCOPE.md: "the actual deliverable" of this slice."""
from __future__ import annotations

from business_credit_e2e import reuse_inventory


def test_report_computes_from_the_data_not_a_narrative():
    indicators = reuse_inventory.report()
    assert indicators["forks"]["value"] == 0
    assert indicators["identity_passthrough_relabels"]["value"] == reuse_inventory.PASSTHROUGH_RELABEL_COUNT
    assert indicators["gaps_resolved_by_compose"]["of_total_gaps"] == len(reuse_inventory.GAP_REGISTER)


def test_no_gap_is_resolved_by_fork():
    """§5.17.2's own rule: fork is never a declared resolution."""
    assert all(g["resolution"] != "fork" for g in reuse_inventory.GAP_REGISTER)


def test_every_gap_names_an_owner_and_a_reason():
    for gap in reuse_inventory.GAP_REGISTER:
        assert gap["owner"]
        assert gap["reason"]


def test_reuse_table_covers_every_component_deps_md_names_as_hard():
    components = {row["component"] for row in reuse_inventory.REUSE_TABLE}
    for name in ("project 00", "project 02", "project 05", "project 06", "project 07"):
        assert any(name in c for c in components), f"{name} missing from REUSE_TABLE"
