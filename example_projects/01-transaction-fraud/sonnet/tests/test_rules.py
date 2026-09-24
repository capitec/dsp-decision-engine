from decider.steps.trees import TreeConfig

from fraud_interdiction import vocab
from fraud_interdiction.rules import build_overlay_base_document, build_rule_catalog, build_tree_document


def test_catalog_matches_spec_6_1_counts():
    catalog = build_rule_catalog()
    assert len(catalog) == 521 + 114
    assert len(catalog.by_status(vocab.STATUS_LIVE)) == 521
    assert len(catalog.by_status(vocab.STATUS_SHADOW)) == 114
    # §6.1's per-family table, verbatim.
    live_by_family = {}
    for r in catalog.by_status(vocab.STATUS_LIVE):
        live_by_family[r.family] = live_by_family.get(r.family, 0) + 1
    assert live_by_family == {"CF": 214, "AT": 112, "MS": 96, "FP": 41, "AA": 58}


def test_rule_ids_are_unique_and_stable_format():
    catalog = build_rule_catalog()
    ids = [r.rule_id for r in catalog.rules]
    assert len(ids) == len(set(ids))
    assert all(len(rid) == 7 and rid[2] == "-" for rid in ids)  # "CF-0001"


def test_generation_is_deterministic():
    """09 §5.15 item 3: a named, fixed seed -- not a call to a random number generator."""
    a = build_rule_catalog()
    b = build_rule_catalog()
    assert [r.condition for r in a.rules] == [r.condition for r in b.rules]
    assert [r.action_code for r in a.rules] == [r.action_code for r in b.rules]


def test_overlay_eligible_rules_have_a_base_condition():
    catalog = build_rule_catalog()
    eligible = catalog.overlay_eligible()
    assert len(eligible) == 18
    assert all(r.family == vocab.FAMILY_MULE_SCAM for r in eligible)
    assert all(r.base_condition is not None for r in eligible)
    assert all(not r.overlay_exempt for r in eligible)  # exempt and eligible would be a contradiction


def test_live_and_overlay_base_documents_load_and_run():
    """Real fit-for-purpose check for TreeConfig at the spec's real rule volume (§10 item 15)."""
    catalog = build_rule_catalog()
    live = catalog.by_status(vocab.STATUS_LIVE)
    live_cfg = TreeConfig.load(build_tree_document(live, "live_rules"))
    base_cfg = TreeConfig.load(build_overlay_base_document(live, "overlay_base"))
    assert len(live_cfg.tree.to_tree().rules) == 521
    assert len(base_cfg.tree.to_tree().rules) == 18


def test_cf_and_fp_rules_never_apply_to_instant_payments():
    """SCOPE.md restricts this slice to instant payments; CF/FP rules exist for volume only (§10.15)."""
    catalog = build_rule_catalog()
    for r in catalog.rules:
        if r.family in (vocab.FAMILY_CARD_FRAUD, vocab.FAMILY_FIRST_PARTY_FRAUD):
            assert vocab.EVENT_TYPE_INSTANT_PAYMENT not in r.event_types
        else:
            assert vocab.EVENT_TYPE_INSTANT_PAYMENT in r.event_types
