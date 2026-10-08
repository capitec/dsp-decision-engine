"""Write `configs/<version>/`: the synthetic rule set, the client wording table and the params document.

    uv run python examples/transaction-fraud/scripts/generate_configs.py [version]

The rules are generated from a fixed seed so the files are reproducible; in a real deployment analysts edit
`rule_set.json` (or its source) directly, and a new version is a new directory.
"""
import json
import random
import sys
from pathlib import Path

from decider import DecisionTableConfig

from fraud_interdiction.pipeline import build
from fraud_interdiction.rules import RuleSet
from fraud_interdiction.vocab import Action

SEED = 20260115
MULTIPLIER = "mule_scam_amount_multiplier"
# Live and shadow rules per family.
COUNTS = {"CF": (214, 31), "AT": (112, 22), "MS": (96, 40), "FP": (41, 9), "AA": (58, 12)}
OVERLAY_ADJUSTED_MS_RULES = 18
SEGMENTS = ("new_to_bank", "high_value", "vulnerable", "offshore_traveller", "prior_fraud_victim")
REASON_BASE = {"CF": 1000, "AT": 2000, "MS": 3000, "FP": 4000, "AA": 5000}


def gt(feature, threshold): return {"op": ">", "feature": feature, "threshold": threshold}
def lt(feature, threshold): return {"op": "<", "feature": feature, "threshold": threshold}
def gte(feature, threshold): return {"op": ">=", "feature": feature, "threshold": threshold}
def is_true(feature): return {"op": "is_true", "feature": feature}
def between(feature, lo, hi): return {"op": "between", "feature": feature, "min": lo, "max": hi}
def computed(expression, threshold): return {"op": ">", "feature": {"type": "computed", "expression": expression},
                                             "threshold": threshold}
def all_of(*c): return {"type": "composite", "op": "and", "conditions": list(c)}
def any_of(*c): return {"type": "composite", "op": "or", "conditions": list(c)}


def amount_over(threshold, adjusted):
    """The amount test, either at the authored threshold or scaled by the overlay multiplier."""
    return computed(f"amount - {threshold} * {MULTIPLIER}", 0.0) if adjusted else gt("amount", threshold)


def new_beneficiary_payment(rng):
    """A first payment to a new, high-mule-rate beneficiary soon after a device change."""
    amount, hours = round(rng.uniform(4000, 15000), -2), rng.choice([1, 2, 4, 6, 12, 24])
    device, band = rng.choice([24, 48, 72, 96]), rng.choice([3, 4, 5])
    return lambda adj: all_of(is_true("beneficiary_first_payment"), lt("beneficiary_age_hours", hours),
                              amount_over(amount, adj), gte("beneficiary_bank_mule_rate_band", band),
                              lt("device_change_hours", device)), None


def rapid_payments(rng):
    count, window = rng.choice([2, 3, 4, 5]), rng.choice(["velocity_count_1min", "velocity_count_10min"])
    amount = round(rng.uniform(2000, 8000), -2)
    return lambda adj: all_of(gte(window, count), amount_over(amount, adj), lt("beneficiary_age_hours", 24)), window


def share_of_day(rng):
    ratio = round(rng.uniform(0.4, 0.9), 2)
    count = rng.choice([1, 2, 3])
    return lambda adj: all_of(computed("amount / (velocity_sum_amount_24h + 1.0)", ratio),
                              gt("velocity_distinct_counterparties_24h", count)), None


def sim_or_device_change(rng):
    sim, device, amount = rng.choice([1, 2, 4, 6, 12]), rng.choice([1, 2, 6, 12, 24]), round(rng.uniform(500, 3000), -1)
    return lambda adj: any_of(all_of(lt("sim_change_hours", sim), gt("amount", amount)),
                              all_of(lt("device_change_hours", device), gte("session_ip_distance_band", 2))), None


def repeat_card_use(rng):
    count, counterparties = rng.choice([3, 4, 5, 6, 8]), rng.choice([2, 3, 4])
    return lambda adj: all_of(gte("velocity_count_1h", count),
                              gt("velocity_distinct_counterparties_24h", counterparties),
                              between("device_reputation_band", 3, 5)), None


def model_score(rng):
    cutoff = rng.choice([620, 660, 700, 740, 780, 820])
    return lambda adj: all_of(is_true("model_score_available"), gt("model_score", cutoff)), None


def prior_fraud_victim(rng):
    amount = round(rng.uniform(1000, 5000), -1)
    return lambda adj: all_of(gt("client_prior_confirmed_fraud_count", 0), is_true("client_vulnerability_flag"),
                              gt("amount", amount)), None


def large_unmatched_payment(rng):
    amount, counterparties = round(rng.uniform(20000, 100000), -2), rng.choice([3, 4, 5])
    return lambda adj: all_of(gt("amount", amount), gte("cop_name_match_band", 1),
                              gte("velocity_distinct_counterparties_24h", counterparties)), None


# About one rule in nine reads the model score.
TEMPLATES = {"MS": (new_beneficiary_payment, rapid_payments, share_of_day),
             "AT": (sim_or_device_change,) * 8 + (model_score,), "CF": (repeat_card_use,) * 8 + (model_score,),
             "FP": (prior_fraud_victim,), "AA": (large_unmatched_payment,)}
EVENT_TYPES = {"MS": [210], "AT": [210], "AA": [210], "CF": [110, 111], "FP": [220]}


def action_for(family, severity, rng):
    if family == "AA" and severity >= 4:
        return Action.DECLINE
    if severity >= 5:
        return rng.choice([Action.DECLINE, Action.HOLD_FOR_REVIEW])
    if severity == 4:
        return rng.choice([Action.HOLD_FOR_REVIEW, Action.STEP_UP])
    return rng.choice([Action.STEP_UP, Action.MONITOR]) if severity == 3 else Action.MONITOR


def rule(rule_id, family, status, severity, action, rng, build_condition, window, adjusted, **extra):
    index = int(rule_id[3:])
    condition = build_condition(adjusted)
    return {
        "rule_id": rule_id, "family": family, "status": status, "severity": severity, "action_code": int(action),
        "priority": rng.randint(0, 999), "reason_code": REASON_BASE[family] + index % 40,
        "critical": family == "AA" and index % 23 == 0, "event_types": EVENT_TYPES[family],
        "segments": None if index % 3 == 0 else rng.sample(SEGMENTS, k=rng.choice([2, 3])),
        "effective_from": "2025-01-01", "effective_to": None, "velocity_feature": window,
        "stale_absent": ("suppress" if index % 3 == 0 else "evaluate_false") if window else "last_known_good",
        "overlay_exempt": not adjusted and (family == "AA" or (family == "MS" and index % 11 == 0)),
        "condition": condition, "base_condition": build_condition(False) if adjusted else None, **extra,
    }


def generate_rules():
    rng = random.Random(SEED)
    rules, adjusted_left = [], OVERLAY_ADJUSTED_MS_RULES
    for family, (n_live, n_shadow) in COUNTS.items():
        templates = TEMPLATES[family]
        for index in range(1, n_live + n_shadow + 1):
            status = "live" if index <= n_live else "shadow"
            template = templates[index % len(templates)]
            adjusted = status == "live" and family == "MS" and template is not share_of_day and adjusted_left > 0
            adjusted_left -= adjusted
            build_condition, window = template(rng)
            severity = rng.choice([1, 2, 3, 3, 4, 5])
            rules.append(rule(f"{family}-{index:04d}", family, status, severity, action_for(family, severity, rng),
                              rng, build_condition, window, adjusted))
    return rules


def ms_0208(rng):
    """The mule rule from the business description: R8 000 first payment to a new beneficiary, device changed in 72 hours."""
    build_condition = lambda adj: all_of(is_true("beneficiary_first_payment"), lt("beneficiary_age_hours", 2),
                                         amount_over(8000, adj), gte("beneficiary_bank_mule_rate_band", 4),
                                         lt("device_change_hours", 72))
    return {**rule("MS-0208", "MS", "live", 4, Action.HOLD_FOR_REVIEW, rng, build_condition, None, True),
            "priority": 100, "critical": False, "segments": None}


WORDING = [(Action.FREEZE_ACCOUNT, "freeze_account"), (Action.BLOCK_CHANNEL, "block_channel"),
           (Action.DECLINE, "decline"), (Action.HOLD_FOR_REVIEW, "hold_for_review"), (Action.STEP_UP, "step_up")]


def wording_table():
    return {
        "type": "decision_table", "name": "client_wording",
        "columns": {"action": "Float64", "client_wording_key": "String", "client_visible": "Boolean"},
        "rows": [{"action": float(a), "client_wording_key": f"wording.{name}.v1", "client_visible": True}
                 for a, name in WORDING],
        "expression": {"type": "eq", "variable": "action_code", "value_column": "action"},
        "outputs": ["client_wording_key", "client_visible"], "default": ["wording.none.v1", False],
    }


def main(version="0.1.0"):
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    rules = generate_rules()
    # MS-0208 stands in for the 96th live mule rule, so the counts still match the estate.
    rules = [r for r in rules if r["rule_id"] != "MS-0096"] + [ms_0208(random.Random(SEED))]
    rule_set = {"type": "rule_set", "name": "rules", "rules": rules}
    (out / "rule_set.json").write_text(json.dumps(rule_set, separators=(",", ":")))
    (out / "client_wording.json").write_text(json.dumps(wording_table(), indent=2) + "\n")
    pipeline = build(RuleSet.load(rule_set), DecisionTableConfig.load(wording_table()))
    (out / "params.json").write_text(json.dumps(pipeline.parameters().defaults(), indent=2) + "\n")
    print(f"wrote {len(rules)} rules to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
