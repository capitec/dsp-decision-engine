import json
from collections import Counter
from pathlib import Path

from fraud_interdiction.rules import RuleSet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "0.1.0"


def load():
    return RuleSet.load(str(CONFIGS / "rule_set.json"))


def test_the_estate_has_the_live_and_shadow_rules_of_each_family():
    rules = load().rules
    live = Counter(r.family for r in rules if r.status == "live")
    shadow = Counter(r.family for r in rules if r.status == "shadow")
    assert live == {"CF": 214, "AT": 112, "MS": 96, "FP": 41, "AA": 58}
    assert shadow == {"CF": 31, "AT": 22, "MS": 40, "FP": 9, "AA": 12}


def test_every_rule_id_is_unique_and_its_reason_code_is_in_the_registry(documents):
    rules = load().rules
    assert len({r.rule_id for r in rules}) == len(rules) == 635
    assert all(r.reason_code in documents["reason_codes"] for r in rules)


def test_ms_0208_is_the_r8000_first_payment_rule_and_reads_the_overlay_multiplier():
    ms_0208 = next(r for r in load().rules if r.rule_id == "MS-0208")
    assert (ms_0208.family, ms_0208.severity, ms_0208.action_code, ms_0208.status) == ("MS", 4, 40, "live")
    assert "amount - 8000 * mule_scam_amount_multiplier" in json.dumps(ms_0208.condition)
    assert {"op": ">", "feature": "amount", "threshold": 8000} in ms_0208.base_condition["conditions"]


def test_only_live_mule_and_scam_rules_read_the_overlay_adjusted_threshold():
    adjusted = [r for r in load().rules if r.base_condition is not None]
    assert 18 <= len(adjusted) <= 20
    assert all(r.family == "MS" and r.status == "live" and not r.overlay_exempt for r in adjusted)


def test_card_and_first_party_rules_are_never_applicable_to_an_instant_payment():
    for r in load().rules:
        assert (210 in r.event_types) == (r.family in ("MS", "AT", "AA")), r.rule_id
