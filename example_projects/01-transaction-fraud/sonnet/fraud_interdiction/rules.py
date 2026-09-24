"""The flat rule set (spec 01 §5.10, §6.1): a generated ~520 live + ~100 shadow rules.

**Is a flat rule set a core component kind (spec 01 §13 Q1)?** decider has
one built-in shaped for "evaluate a population of independent, named
rules and keep every result": `decider.steps.trees.TreeConfig` with a
`prioritized_flat_rule` document and `mode="all"` -- every rule is
evaluated (no early exit), and each rule's leaf output becomes its own
`<rule_id>.<column>` output column. That gives us the rule *shape*
(predicates -> true/false) at the framework's own vectorised, kernel-run
speed, tested here at real volume (635 rules -- see
`tests/test_rules.py::test_live_and_overlay_base_documents_load_and_run`
and NOTES.md "Framework friction" §4.2).

What `TreeConfig` does **not** carry is everything spec 01 calls
"governance metadata that is not part of the answer": owner, family,
severity, action, priority, status, effective dates, segments,
overlay-exemption, criticality, reason code, queue, stale/absent
behaviour. A tree node has no field for any of it, and it should not --
that data does not vary per record, so putting it in the tree (or in 620
constant-valued output columns) would make every one of 09 §5.15 items
1-2's identity/governance questions a data problem instead of a lookup.

So this module keeps two things side by side, exactly as the spec
describes the rule set (§6.1's attribute table vs. its "Predicates" row):

- `RuleDefinition`: the governance record, one per `rule_id`, a plain
  Python object (`RuleCatalog`), the same "registry, not a calculation"
  shape as `credit_core.reason_codes.ReasonCodeRegistry`.
- the flat-rule JSON document generated from those definitions' shapes,
  loaded into a `TreeConfig` that answers only "did this predicate hold".

`firing.py` joins them back together into `fired_rule_ids` with per-rule
detail, which is where "the answer is a set with an argmax over it, not a
lookup" (§2 Q5) actually gets computed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from fraud_interdiction import vocab

RULE_SET_VERSION = 1
RULE_GENERATION_SEED = 20260115  # fixed and named (09 §5.15 item 3): generation is deterministic, not "random".

# A handful of the estate's 46 segments (§5.8); "all" (segments=None) covers the rest.
SEGMENT_NEW_TO_BANK = "new_to_bank"
SEGMENT_HIGH_VALUE = "high_value"
SEGMENT_VULNERABLE = "vulnerable"
SEGMENT_OFFSHORE_TRAVELLER = "offshore_traveller"
SEGMENT_PRIOR_FRAUD_VICTIM = "prior_fraud_victim"
ALL_SEGMENTS = (SEGMENT_NEW_TO_BANK, SEGMENT_HIGH_VALUE, SEGMENT_VULNERABLE,
                SEGMENT_OFFSHORE_TRAVELLER, SEGMENT_PRIOR_FRAUD_VICTIM)

# The overlay target every MS amount-threshold-eligible rule's condition references (overlays.py).
MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE = "mule_scam_amount_multiplier"


@dataclass(frozen=True)
class RuleDefinition:
    """One rule's governance record (§6.1's attribute table). Immutable -- a change is a new `rule_version`."""

    rule_id: str
    rule_version: int
    family: str
    severity: int  # 1-5
    action_code: int
    priority: int  # 0-999 within action
    status: str  # live | shadow | retired
    effective_from: date
    effective_to: date | None
    event_types: tuple[int, ...]
    segments: tuple[str, ...] | None  # None == "all"
    overlay_exempt: bool
    critical: bool
    suppressible: bool
    reason_code: int
    queue: str | None
    stale_absent_behavior: str
    max_fire_rate_pct: float
    condition: dict  # the flat-rule "condition" node this rule's predicates compile to
    predicate_count: int
    overlay_eligible: bool  # True if `condition` references MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE
    base_condition: dict | None  # same shape, literal (unadjusted) threshold -- only set if overlay_eligible
    references_velocity: bool
    velocity_feature: str | None  # which of the 3 state-tracked aggregates (features.py), if any
    description_internal: str
    description_client_safe: str


class RuleCatalog:
    """Every live, shadow and (declared but unused here) retired rule, keyed by `rule_id`.

    Testable standalone, no pipeline needed -- same shape as
    `credit_core.reason_codes.ReasonCodeRegistry`.
    """

    def __init__(self, rules: list[RuleDefinition]):
        by_id = {r.rule_id: r for r in rules}
        if len(by_id) != len(rules):
            ids = [r.rule_id for r in rules]
            dupes = sorted({rid for rid in ids if ids.count(rid) > 1})
            raise ValueError(f"duplicate rule_id(s): {dupes}")
        self._by_id = by_id
        self.rules = tuple(rules)

    def __getitem__(self, rule_id: str) -> RuleDefinition:
        return self._by_id[rule_id]

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def by_status(self, status: str) -> tuple[RuleDefinition, ...]:
        return tuple(r for r in self.rules if r.status == status)

    def overlay_eligible(self) -> tuple[RuleDefinition, ...]:
        return tuple(r for r in self.rules if r.overlay_eligible)


# --- Predicate builders (flat-rule condition JSON) ---------------------------

def _unary(op: str, feature: str, **kw) -> dict:
    return {"op": op, "feature": feature, **kw}


def _gt(feature: str, threshold: float) -> dict:
    return _unary(">", feature, threshold=threshold)


def _lt(feature: str, threshold: float) -> dict:
    return _unary("<", feature, threshold=threshold)


def _gte(feature: str, threshold: float) -> dict:
    return _unary(">=", feature, threshold=threshold)


def _between(feature: str, lo: float, hi: float) -> dict:
    return {"op": "between", "feature": feature, "min": lo, "max": hi}


def _is_true(feature: str) -> dict:
    return {"op": "is_true", "feature": feature}


def _computed_gt(expression: str, threshold: float = 0.0) -> dict:
    return {"op": ">", "feature": {"type": "computed", "expression": expression}, "threshold": threshold}


def _and(*conditions: dict) -> dict:
    conditions = [c for c in conditions if c is not None]
    return conditions[0] if len(conditions) == 1 else {"type": "composite", "op": "and", "conditions": conditions}


def _or(*conditions: dict) -> dict:
    return {"type": "composite", "op": "or", "conditions": conditions}


# --- Rule shape templates -----------------------------------------------------
# Each template returns (condition, base_condition_or_None, predicate_count, references_velocity,
# overlay_eligible). Thresholds are jittered per-instance by the generator so ~500 rules aren't
# identical copies, matching "generated" (SCOPE.md), not hand-authored.

def _mule_amount_template(rng: random.Random, overlay_eligible: bool):
    """§5.10's own worked example (MS-0208): first payment, recent beneficiary, amount, device change."""
    amount_thresh = round(rng.uniform(4000, 15000), -2)
    age_hours = rng.choice([1, 2, 4, 6, 12, 24])
    device_hours = rng.choice([24, 48, 72, 96])
    mule_band = rng.choice([3, 4, 5])
    amount_cond = (
        _computed_gt(f"amount - {amount_thresh} * {MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE}")
        if overlay_eligible else _gt("amount", amount_thresh)
    )
    condition = _and(
        _is_true("beneficiary_first_payment"),
        _lt("beneficiary_age_hours", age_hours),
        amount_cond,
        _gte("beneficiary_bank_mule_rate_band", mule_band),
        _lt("device_change_hours", device_hours),
    )
    base = _and(
        _is_true("beneficiary_first_payment"), _lt("beneficiary_age_hours", age_hours),
        _gt("amount", amount_thresh), _gte("beneficiary_bank_mule_rate_band", mule_band),
        _lt("device_change_hours", device_hours),
    ) if overlay_eligible else None
    return condition, base, 5, False, overlay_eligible, None


def _mule_velocity_template(rng: random.Random, overlay_eligible: bool):
    count_thresh = rng.choice([2, 3, 4, 5])
    window = rng.choice(["velocity_count_1min", "velocity_count_10min"])
    amount_thresh = round(rng.uniform(2000, 8000), -2)
    amount_cond = (
        _computed_gt(f"amount - {amount_thresh} * {MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE}")
        if overlay_eligible else _gt("amount", amount_thresh)
    )
    condition = _and(_gte(window, count_thresh), amount_cond, _lt("beneficiary_age_hours", 24))
    base = _and(_gte(window, count_thresh), _gt("amount", amount_thresh),
                _lt("beneficiary_age_hours", 24)) if overlay_eligible else None
    return condition, base, 3, True, overlay_eligible, window


def _account_takeover_template(rng: random.Random, **_):
    sim_hours = rng.choice([1, 2, 4, 6, 12])
    device_hours = rng.choice([1, 2, 6, 12, 24])
    condition = _or(
        _and(_lt("sim_change_hours", sim_hours), _gt("amount", round(rng.uniform(500, 3000), -1))),
        _and(_lt("device_change_hours", device_hours), _gte("session_ip_distance_band", 2)),
    )
    return condition, None, 4, False, False, None


def _card_fraud_velocity_template(rng: random.Random, **_):
    """Card-not-present shape (event_types will exclude 210, see `_family_event_types`)."""
    count_thresh = rng.choice([3, 4, 5, 6, 8])
    condition = _and(
        _gte("velocity_count_1h", count_thresh),
        _gt("velocity_distinct_counterparties_24h", rng.choice([2, 3, 4])),
        _between("device_reputation_band", 3, 5),
    )
    return condition, None, 3, True, False, None


def _model_score_template(rng: random.Random, **_):
    cutoff = rng.choice([540, 570, 600, 620, 650, 700])
    condition = _and(_is_true("model_score_available"), _gt("model_score", cutoff))
    return condition, None, 2, False, False, None


def _derived_ratio_template(rng: random.Random, **_):
    """Rule-local derived quantity (§5.10: 3% of the live set), a `ComputedFeature` ratio."""
    ratio_thresh = round(rng.uniform(0.4, 0.9), 2)
    condition = _and(
        {"op": ">", "feature": {"type": "computed", "expression": "amount / (velocity_sum_amount_24h + 1.0)"},
         "threshold": ratio_thresh},
        _gt("velocity_distinct_counterparties_24h", rng.choice([1, 2, 3])),
    )
    return condition, None, 2, True, False, None


def _first_party_template(rng: random.Random, **_):
    condition = _and(
        _gt("client_prior_confirmed_fraud_count", 0),
        _is_true("client_vulnerability_flag"),
        _gt("amount", round(rng.uniform(1000, 5000), -1)),
    )
    return condition, None, 3, False, False, None


def _aml_adjacent_template(rng: random.Random, **_):
    condition = _and(
        _gt("amount", round(rng.uniform(20000, 100000), -2)),
        _gte("cop_name_match_band", vocab.ConfirmationOfPayeeBand.NO_MATCH.value),
        _gte("velocity_distinct_counterparties_24h", rng.choice([3, 4, 5])),
    )
    return condition, None, 3, False, False, None


_FAMILY_TEMPLATES = {
    vocab.FAMILY_MULE_SCAM: (_mule_amount_template, _mule_velocity_template, _derived_ratio_template),
    vocab.FAMILY_ACCOUNT_TAKEOVER: (_account_takeover_template, _model_score_template),
    vocab.FAMILY_CARD_FRAUD: (_card_fraud_velocity_template, _model_score_template),
    vocab.FAMILY_FIRST_PARTY_FRAUD: (_first_party_template,),
    vocab.FAMILY_AML_ADJACENT: (_aml_adjacent_template,),
}

# §6.1 family x live/shadow counts, verbatim from the spec table.
_FAMILY_COUNTS = {
    vocab.FAMILY_CARD_FRAUD: (214, 31),
    vocab.FAMILY_ACCOUNT_TAKEOVER: (112, 22),
    vocab.FAMILY_MULE_SCAM: (96, 40),
    vocab.FAMILY_FIRST_PARTY_FRAUD: (41, 9),
    vocab.FAMILY_AML_ADJACENT: (58, 12),
}

_MS_OVERLAY_ELIGIBLE_LIVE = 18  # of 96 live MS rules, how many carry an overlay-adjustable amount threshold


def _family_event_types(family: str) -> tuple[int, ...]:
    """Only MS, AT and AA rules apply to instant payments in this slice's scope (SCOPE.md).

    CF and FP rules exist (governance volume, applicable-population cost, §10 item 15) but their
    real event types (card authorisation, debit-order dispute) are out of scope here, so they are
    correctly never in the *applicable* population for an instant payment (§5.8) -- evaluated
    (cheap, vectorised) but always filtered out, exactly like the ~599 rules that evaluate false
    on any one real event.
    """
    if family in (vocab.FAMILY_MULE_SCAM, vocab.FAMILY_ACCOUNT_TAKEOVER, vocab.FAMILY_AML_ADJACENT):
        return (vocab.EVENT_TYPE_INSTANT_PAYMENT,)
    return (110, 111) if family == vocab.FAMILY_CARD_FRAUD else (220,)


def _reason_code_for(family: str, index: int) -> int:
    base = {"CF": 1000, "AT": 2000, "MS": 3000, "FP": 4000, "AA": 5000}[family]
    return base + (index % 40)  # ~40 distinct wordings per family, not one per rule (§5.14)


def build_rule_catalog(seed: int = RULE_GENERATION_SEED) -> RuleCatalog:
    """Generate the estate: 521 live + 114 shadow rules across 5 families (§6.1's exact counts)."""
    rng = random.Random(seed)
    rules: list[RuleDefinition] = []
    ms_overlay_eligible_used = 0
    for family, (n_live, n_shadow) in _FAMILY_COUNTS.items():
        templates = _FAMILY_TEMPLATES[family]
        event_types = _family_event_types(family)
        next_index = 1
        for status, n in ((vocab.STATUS_LIVE, n_live), (vocab.STATUS_SHADOW, n_shadow)):
            for _ in range(n):
                i = next_index
                next_index += 1
                template = templates[i % len(templates)]
                overlay_eligible = (
                    family == vocab.FAMILY_MULE_SCAM and status == vocab.STATUS_LIVE
                    and template is _mule_amount_template
                    and ms_overlay_eligible_used < _MS_OVERLAY_ELIGIBLE_LIVE
                )
                condition, base_condition, n_pred, ref_velocity, is_overlay_eligible, velocity_feature = template(
                    rng, overlay_eligible=overlay_eligible)
                if is_overlay_eligible:
                    ms_overlay_eligible_used += 1
                severity = rng.choice([1, 2, 3, 3, 4, 5])
                action_code = _action_for(family, severity, rng)
                # Overlay-exempt and overlay-eligible are contradictory (§6.5: "rules marked
                # overlay-exempt cannot have their thresholds modified by any overlay") --
                # `is_overlay_eligible` rules are excluded from exemption below.
                overlay_exempt = not is_overlay_eligible and (
                    family == vocab.FAMILY_AML_ADJACENT or (family == vocab.FAMILY_MULE_SCAM and i % 11 == 0))
                critical = family == vocab.FAMILY_AML_ADJACENT and i % 23 == 0
                segments = None if i % 3 == 0 else tuple(rng.sample(ALL_SEGMENTS, k=rng.choice([2, 3])))
                rule_id = f"{family}-{i:04d}"
                rules.append(RuleDefinition(
                    rule_id=rule_id, rule_version=1, family=family, severity=severity, action_code=action_code,
                    priority=rng.randint(0, 999), status=status,
                    effective_from=date(2025, 1, 1), effective_to=None,
                    event_types=event_types, segments=segments,
                    overlay_exempt=overlay_exempt, critical=critical,
                    suppressible=velocity_feature is not None, reason_code=_reason_code_for(family, i),
                    queue=f"{family}-1" if action_code == vocab.ACTION_HOLD_FOR_REVIEW else None,
                    stale_absent_behavior=(
                        vocab.STALE_ABSENT_SUPPRESS if velocity_feature is not None and i % 3 == 0
                        else vocab.STALE_ABSENT_EVALUATE_FALSE if velocity_feature is not None
                        else vocab.STALE_ABSENT_LAST_KNOWN
                    ),
                    max_fire_rate_pct=5.0,
                    condition=condition, predicate_count=n_pred,
                    overlay_eligible=is_overlay_eligible, base_condition=base_condition,
                    references_velocity=ref_velocity, velocity_feature=velocity_feature,
                    description_internal=f"{vocab.FAMILY_NAMES[family]} rule {rule_id}, generated template "
                                          f"{template.__name__}",
                    description_client_safe="A recent transaction needed extra review.",
                ))
    return RuleCatalog(rules)


def _action_for(family: str, severity: int, rng: random.Random) -> int:
    if family == vocab.FAMILY_AML_ADJACENT and severity >= 4:
        return vocab.ACTION_DECLINE
    if severity >= 5:
        return rng.choice([vocab.ACTION_DECLINE, vocab.ACTION_HOLD_FOR_REVIEW])
    if severity == 4:
        return rng.choice([vocab.ACTION_HOLD_FOR_REVIEW, vocab.ACTION_STEP_UP])
    if severity == 3:
        return rng.choice([vocab.ACTION_STEP_UP, vocab.ACTION_MONITOR])
    return vocab.ACTION_MONITOR


# --- Flat-rule document assembly ---------------------------------------------

def _rule_root(rule: RuleDefinition, condition: dict) -> dict:
    """`condition` is either a raw unary condition (`_gt`/`_lt`/...) or an already rule-node-shaped
    composite from `_and`/`_or` (which carries its own `type: "composite"`); either way it becomes
    one rule root with a single `then` leaf (no `otherwise` -- a non-match selects the default row)."""
    leaf = {"type": "leaf", "result_idx": 0}
    node = {**condition, "then": leaf} if condition.get("type") == "composite" \
        else {"type": "unary", "condition": condition, "then": leaf}
    return {"meta": {"name": rule.rule_id}, "rule": node}


def build_tree_document(rules: tuple[RuleDefinition, ...], name: str) -> dict:
    """A `{"type": "tree", "tree": {"type": "prioritized_flat_rule", "mode": "all", ...}}` document.

    Every rule's `<rule_id>.fired` output column; `mode="all"` means every rule is evaluated with
    no early exit (§5.10), evaluation order does not affect the result (rules are independent).
    """
    return {
        "type": "tree", "name": name,
        "tree": {
            "type": "prioritized_flat_rule", "mode": "all",
            "rules": [_rule_root(r, r.condition) for r in rules],
            "output": {"data": [{"fired": True}], "default": {"fired": False}, "dtypes": [("fired", "Boolean")]},
        },
    }


def build_overlay_base_document(rules: tuple[RuleDefinition, ...], name: str) -> dict:
    """The base (unadjusted) evaluation of overlay-eligible rules only -- for `fired_on_overlay_ids`.

    Output column `fired_base`, not `fired`: this tree's rules share `rule_id`s with `live_rules`
    (they're the same rules, evaluated a second time at their literal threshold), so a `dag` sharing
    both trees would otherwise see two nodes writing the same `<rule_id>.fired` column -- decider
    catches this at bind time (`WiringError: ... both write '<id>.fired'; use flow(...) to apply
    them in written order`), which is correct: `flow`'s "later write wins" semantics would silently
    discard the base evaluation, which is exactly the value this document exists to keep.
    """
    eligible = [r for r in rules if r.overlay_eligible]
    return {
        "type": "tree", "name": name,
        "tree": {
            "type": "prioritized_flat_rule", "mode": "all",
            "rules": [_rule_root(r, r.base_condition) for r in eligible],
            "output": {"data": [{"fired_base": True}], "default": {"fired_base": False},
                       "dtypes": [("fired_base", "Boolean")]},
        },
    }
