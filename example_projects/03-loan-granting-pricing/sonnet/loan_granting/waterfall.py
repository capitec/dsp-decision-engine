"""Stage 5.5 -- the cap waterfall (spec 03 §5.5). 52 rules, three ceilings,
reduce-only with one authority-bounded exception, and the full attribution
chain (which rule bound each ceiling, the complete chain of values, and
which rules were evaluated-not-applicable vs. evaluated-and-did-not-bind).

Written as one plain, deterministic Python function over the rule
register (a list of dicts -- config-shaped data, not code: "adding one rule
... is a configuration change reviewed by its owner, not a release", spec
03 §10 item 15), wrapped in a single `frame_step`. Not `decider.loop`: the
register mixes several genuinely different rule *kinds* (grade-grid
lookups, threshold reductions, a dual-ceiling rule, one authority-bounded
uplift) whose per-iteration logic differs by kind, and the correctness bar
here (§10 item 5: the final value, the binding rule and the full chain must
all be independently retrievable) is easier to get right and keep right as
one reviewable function than as several `loop` steps threaded through
carried state -- see NOTES.md "Framework friction" for the fuller
reasoning, which is the same judgement made for the solve (`solve.py`).

13 rules carry the real conditions from spec 03 §5.5's own table (plus a
pensioner cap and a regulatory sanity ceiling this project adds, both
noted below); 39 more are generated with narrow, rarely-firing conditions
across the same four owners, to keep the register at its declared real
volume (40-60, spec's own range) without 39 rules of invented significance
diluting the ones that matter. Ownership totals match spec 03 §5.5's own
split: Credit Risk Policy 31, Unsecured Lending Product 12, Credit Systems
6, Financial Crime 3 (CAP-0420's "Credit Committee" row in spec's table is
folded into Product's count here -- see NOTES.md "Spec problems").
"""
from __future__ import annotations

from decider import frame_step

from loan_granting import reasons

CEILING_AMOUNT = "amount"
CEILING_TERM = "term"
CEILING_GRADE = "grade"

OWNER_CRP = "Credit Risk Policy"
OWNER_PRODUCT = "Unsecured Lending Product"
OWNER_SYSTEMS = "Credit Systems"
OWNER_FINCRIME = "Financial Crime"

STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_NO_BIND = "evaluated_did_not_bind"
STATUS_BOUND = "bound"
STATUS_COINCIDENT = "coincident"
STATUS_RAISED = "raised"
STATUS_DECLINED = "declined"

SEED_AMOUNT_CAP = 500_000.0
SEED_TERM_CAP = 84.0
SEED_WORST_GRADE = 12

UPLIFT_MAX_PCT = 0.25
UPLIFT_AUTHORITY_CEILING = 250_000.0

# grade -> amount ceiling (a small appetite grid; matches spec's illustrative points).
_APPETITE_AMOUNT_BY_GRADE = {1: 500_000, 2: 450_000, 3: 400_000, 4: 350_000, 5: 280_000, 6: 210_000,
                             7: 150_000, 8: 100_000, 9: 65_000, 10: 45_000, 11: 30_000, 12: 0}
_APPETITE_TERM_BY_GRADE = {g: 84 for g in range(1, 5)} | {g: 72 for g in range(5, 8)} | \
                          {g: 60 for g in range(8, 10)} | {g: 36 for g in range(10, 13)}

_EMPLOYER_WATCHLIST = {90001, 90002, 90003}


def _named_rules() -> list[dict]:
    return [
        {"rule_id": "CAP-0010", "owner": OWNER_PRODUCT, "klass": "product", "sequence": 1, "ceiling": CEILING_AMOUNT,
         "kind": "seed", "value": SEED_AMOUNT_CAP},
        {"rule_id": "CAP-0025", "owner": OWNER_PRODUCT, "klass": "product", "sequence": 2, "ceiling": CEILING_AMOUNT,
         "kind": "channel_closed", "closed_channels": {6}},
        {"rule_id": "CAP-0100", "owner": OWNER_CRP, "klass": "appetite", "sequence": 3, "ceiling": CEILING_AMOUNT,
         "kind": "grade_amount"},
        {"rule_id": "CAP-0150", "owner": OWNER_CRP, "klass": "policy", "sequence": 6, "ceiling": CEILING_AMOUNT,
         "kind": "pensioner_cap"},
        {"rule_id": "CAP-0140", "owner": OWNER_PRODUCT, "klass": "policy", "sequence": 9, "ceiling": CEILING_AMOUNT,
         "kind": "new_to_bank_cap", "cap": 60_000.0},
        {"rule_id": "CAP-0175", "owner": OWNER_CRP, "klass": "policy", "sequence": 12, "ceiling": CEILING_AMOUNT,
         "kind": "employment_tenure_cap"},
        {"rule_id": "CAP-0210", "owner": OWNER_CRP, "klass": "policy", "sequence": 17, "ceiling": (CEILING_AMOUNT, CEILING_GRADE),
         "kind": "arrears_history"},
        {"rule_id": "CAP-0240", "owner": OWNER_CRP, "klass": "policy", "sequence": 21, "ceiling": CEILING_AMOUNT,
         "kind": "enquiry_velocity"},
        {"rule_id": "CAP-0118", "owner": OWNER_CRP, "klass": "appetite", "sequence": 24, "ceiling": CEILING_TERM,
         "kind": "grade_term"},
        {"rule_id": "CAP-0305", "owner": OWNER_SYSTEMS, "klass": "exposure", "sequence": 26, "ceiling": CEILING_AMOUNT,
         "kind": "employer_watchlist", "cap": 50_000.0},
        {"rule_id": "CAP-0330", "owner": OWNER_PRODUCT, "klass": "policy", "sequence": 33, "ceiling": (CEILING_AMOUNT, CEILING_TERM),
         "kind": "channel_cap"},
        {"rule_id": "CAP-0361", "owner": OWNER_SYSTEMS, "klass": "exposure", "sequence": 38, "ceiling": CEILING_AMOUNT,
         "kind": "exposure_headroom"},
        {"rule_id": "CAP-0420", "owner": OWNER_PRODUCT, "klass": "campaign", "sequence": 52, "ceiling": CEILING_AMOUNT,
         "kind": "campaign_uplift"},
    ]


# (field, op, rare_value) tuples used to build filler rules that rarely fire, so the
# register carries real volume without changing the outcome for a typical applicant.
_FILLER_CONDITIONS = [
    ("purpose_code", "eq", 13), ("purpose_code", "eq", 14), ("residency_code", "eq", 3),
    ("channel_code", "eq", 6), ("requested_term_months", "gte", 80), ("campaign_id", "is_null", None),
    ("applicant_age_years", "gte", 70), ("months_employed", "lte", 2),
]
_FILLER_OWNERS = [OWNER_CRP] * 25 + [OWNER_PRODUCT] * 7 + [OWNER_SYSTEMS] * 4 + [OWNER_FINCRIME] * 3
_FILLER_CLASSES = {OWNER_CRP: "policy", OWNER_PRODUCT: "product", OWNER_SYSTEMS: "exposure", OWNER_FINCRIME: "policy"}


_FIRST_FINCRIME_FILLER_INDEX = _FILLER_OWNERS.index(OWNER_FINCRIME)


def _filler_rules(positions: list[int]) -> list[dict]:
    rules = []
    for i, seq in enumerate(positions):
        owner = _FILLER_OWNERS[i % len(_FILLER_OWNERS)]
        field, op, value = _FILLER_CONDITIONS[i % len(_FILLER_CONDITIONS)]
        is_regulatory = i == _FIRST_FINCRIME_FILLER_INDEX
        # The Financial Crime owner's first filler rule is a regulatory-class sanity
        # ceiling on *amount* (never term/grade), so `campaign_uplift`'s "never exceed a
        # regulatory rule" clause compares like with like.
        ceiling = CEILING_AMOUNT if is_regulatory else (CEILING_TERM if i % 6 == 0 else CEILING_AMOUNT)
        klass = "regulatory" if is_regulatory else _FILLER_CLASSES[owner]
        rule_id = f"CAP-{1000 + seq:04d}"
        rules.append({
            "rule_id": rule_id, "owner": owner, "klass": klass, "sequence": seq, "ceiling": ceiling,
            "kind": "threshold_reduce", "field": field, "op": op, "value": value,
            "pct": 0.85 if ceiling == CEILING_AMOUNT else None,
            "abs_cap": 60.0 if ceiling == CEILING_TERM else None,
        })
    return rules


def build_register() -> list[dict]:
    named = _named_rules()
    used = {r["sequence"] for r in named}
    filler_positions = [s for s in range(1, 53) if s not in used]
    rules = named + _filler_rules(filler_positions)
    rules.sort(key=lambda r: r["sequence"])
    assert len(rules) == 52, f"cap register must carry 52 rules, has {len(rules)}"
    return rules


REGISTER = build_register()


def _applicable(rule: dict, app: dict) -> bool:
    kind = rule["kind"]
    if kind == "seed":
        return True
    if kind == "channel_closed":
        return app.get("channel_code") in rule["closed_channels"]
    if kind == "grade_amount" or kind == "grade_term":
        return app.get("risk_grade") is not None
    if kind == "pensioner_cap":
        return app.get("employment_type_code") == 4
    if kind == "new_to_bank_cap":
        return not app.get("internal_tenure_months")
    if kind == "employment_tenure_cap":
        return app.get("months_employed") is not None
    if kind == "arrears_history":
        return app.get("worst_arrears_months") is not None
    if kind == "enquiry_velocity":
        return app.get("bureau_enquiry_count_60d") is not None
    if kind == "employer_watchlist":
        return app.get("employer_id") in _EMPLOYER_WATCHLIST
    if kind == "channel_cap":
        return app.get("channel_code") in (5, 6)
    if kind == "exposure_headroom":
        return app.get("group_exposure_limit") is not None
    if kind == "campaign_uplift":
        return bool(app.get("campaign_id")) and app.get("campaign_is_preapproved", False)
    if kind == "threshold_reduce":
        field, op, value = rule["field"], rule["op"], rule["value"]
        actual = app.get(field)
        if op == "is_null":
            return actual is None
        if actual is None:
            return False
        if op == "eq":
            return actual == value
        if op == "gte":
            return actual >= value
        if op == "lte":
            return actual <= value
    return False


def _candidate(rule: dict, app: dict, current_amount: float, current_term: float, current_grade: int):
    """Returns (updates: dict[ceiling, value], declines: bool, restrained_by: str|None)."""
    kind = rule["kind"]
    if kind == "seed":
        return {CEILING_AMOUNT: rule["value"]}, False, None
    if kind == "channel_closed":
        return {}, True, None
    if kind == "grade_amount":
        grade = app["risk_grade"]
        amt = _APPETITE_AMOUNT_BY_GRADE.get(grade, 0.0)
        return {CEILING_AMOUNT: float(amt)}, amt <= 0.0, None
    if kind == "grade_term":
        grade = app["risk_grade"]
        return {CEILING_TERM: float(_APPETITE_TERM_BY_GRADE.get(grade, 36))}, False, None
    if kind == "pensioner_cap":
        return {CEILING_AMOUNT: 80_000.0}, False, None
    if kind == "new_to_bank_cap":
        return {CEILING_AMOUNT: rule["cap"]}, False, None
    if kind == "employment_tenure_cap":
        months = app.get("months_employed") or 0.0
        if months < 3.0:
            return {}, True, None
        if months < 6.0:
            return {CEILING_AMOUNT: 25_000.0}, False, None
        if months < 24.0:
            return {CEILING_AMOUNT: 70_000.0}, False, None
        return {CEILING_AMOUNT: current_amount}, False, None  # no reduction
    if kind == "arrears_history":
        worst = app.get("worst_arrears_months") or 0
        count = app.get("accounts_in_arrears_count") or 0
        if worst >= 3 and count >= 1:
            return {CEILING_AMOUNT: 40_000.0, CEILING_GRADE: 8}, False, None
        if worst >= 2:
            return {CEILING_AMOUNT: 120_000.0, CEILING_GRADE: 9}, False, None
        return {}, False, None
    if kind == "enquiry_velocity":
        n = app.get("bureau_enquiry_count_60d") or 0
        if n >= 10:
            return {}, True, None
        if n >= 6:
            return {CEILING_AMOUNT: 30_000.0}, False, None
        return {CEILING_AMOUNT: current_amount}, False, None
    if kind == "employer_watchlist":
        return {CEILING_AMOUNT: rule["cap"]}, False, None
    if kind == "channel_cap":
        if app.get("channel_code") == 6:
            return {CEILING_AMOUNT: 150_000.0, CEILING_TERM: 60.0}, False, None
        return {CEILING_AMOUNT: 100_000.0}, False, None
    if kind == "exposure_headroom":
        headroom = max(0.0, (app.get("group_exposure_limit") or 0.0) - (app.get("internal_exposure_total") or 0.0))
        return {CEILING_AMOUNT: headroom}, False, None
    if kind == "campaign_uplift":
        authorised = min(current_amount * (1.0 + UPLIFT_MAX_PCT), UPLIFT_AUTHORITY_CEILING)
        return {CEILING_AMOUNT: authorised}, False, "uplift"
    if kind == "threshold_reduce":
        if rule["ceiling"] == CEILING_TERM:
            return {CEILING_TERM: min(current_term, rule["abs_cap"])}, False, None
        return {CEILING_AMOUNT: current_amount * rule["pct"]}, False, None
    return {}, False, None


def _run_waterfall(app: dict) -> dict:
    amount_cap, term_cap, worst_grade = SEED_AMOUNT_CAP, SEED_TERM_CAP, SEED_WORST_GRADE
    binder = {CEILING_AMOUNT: None, CEILING_TERM: None, CEILING_GRADE: None}
    chain_ids: dict[str, list[str]] = {CEILING_AMOUNT: [], CEILING_TERM: [], CEILING_GRADE: []}
    chain_values: dict[str, list[float]] = {CEILING_AMOUNT: [], CEILING_TERM: [], CEILING_GRADE: []}
    regulatory_min = None

    rule_ids, rule_owners, rule_ceilings, rule_status, rule_before, rule_after = [], [], [], [], [], []
    decline_reasons: list[int] = []
    uplift_authority_reference = None
    uplift_restrained_by = None

    for rule in REGISTER:
        applicable = _applicable(rule, app)
        rule_ids.append(rule["rule_id"])
        rule_owners.append(rule["owner"])
        ceiling_label = rule["ceiling"] if isinstance(rule["ceiling"], str) else "+".join(rule["ceiling"])
        rule_ceilings.append(ceiling_label)

        if not applicable:
            rule_status.append(STATUS_NOT_APPLICABLE)
            rule_before.append(None)
            rule_after.append(None)
            continue

        current = {CEILING_AMOUNT: amount_cap, CEILING_TERM: term_cap, CEILING_GRADE: worst_grade}
        before_summary = amount_cap if rule["ceiling"] == CEILING_AMOUNT or CEILING_AMOUNT in (rule["ceiling"] or ()) \
            else (term_cap if rule["ceiling"] == CEILING_TERM else worst_grade)
        updates, declines, restrained_kind = _candidate(rule, app, amount_cap, term_cap, worst_grade)

        if declines:
            rule_status.append(STATUS_DECLINED)
            rule_before.append(float(before_summary))
            rule_after.append(float(before_summary))
            decline_reasons.append({
                "CAP-0025": reasons.R_CHANNEL_CLOSED, "CAP-0100": reasons.R_GRADE_BEYOND_APPETITE,
                "CAP-0175": reasons.R_EMPLOYMENT_TENURE_TOO_SHORT, "CAP-0240": reasons.R_ENQUIRY_VELOCITY_EXCESSIVE,
            }.get(rule["rule_id"], reasons.R_GRADE_BEYOND_APPETITE))
            continue

        bound_any = False
        after_summary = before_summary
        for ceiling, candidate in updates.items():
            before = current[ceiling]
            if ceiling == CEILING_AMOUNT and restrained_kind == "uplift":
                # The one raise: bounded above by any regulatory-class value already bound.
                ceiling_value = candidate
                if regulatory_min is not None and ceiling_value > regulatory_min:
                    ceiling_value = regulatory_min
                    uplift_restrained_by = binder[CEILING_AMOUNT]
                uplift_authority_reference = "CC-2026-14"
                amount_cap = ceiling_value
                binder[CEILING_AMOUNT] = rule["rule_id"]
                chain_ids[CEILING_AMOUNT].append(rule["rule_id"])
                chain_values[CEILING_AMOUNT].append(amount_cap)
                bound_any = True
                after_summary = amount_cap
                continue
            if candidate < before:
                if ceiling == CEILING_AMOUNT:
                    amount_cap = candidate
                elif ceiling == CEILING_TERM:
                    term_cap = candidate
                else:
                    worst_grade = candidate
                binder[ceiling] = rule["rule_id"]
                chain_ids[ceiling].append(rule["rule_id"])
                chain_values[ceiling].append(candidate)
                if rule["klass"] == "regulatory":
                    regulatory_min = candidate if regulatory_min is None else min(regulatory_min, candidate)
                bound_any = True
                after_summary = candidate
            # candidate >= before: evaluated, did not bind (spec's "coincident" sub-case --
            # a later rule's candidate exactly equalling the current value -- is not
            # distinguished from an ordinary non-bind here; see NOTES.md "What I left out").

        if restrained_kind == "uplift":
            rule_status.append(STATUS_RAISED)
        elif bound_any:
            rule_status.append(STATUS_BOUND)
        else:
            rule_status.append(STATUS_NO_BIND)
        # `before_summary`/`after_summary` -- unlike each ceiling's own chain list -- mix
        # amount/term (float) and grade (int) values across rows of one flat list column,
        # which polars requires to be one consistent dtype; cast explicitly rather than let
        # it fail wherever the first int (a grade) happens to appear.
        rule_before.append(float(before_summary))
        rule_after.append(float(after_summary))

    return {
        "amount_cap": round(amount_cap, 2), "term_cap": term_cap, "worst_acceptable_grade": worst_grade,
        "amount_cap_binding_rule_id": binder[CEILING_AMOUNT], "term_cap_binding_rule_id": binder[CEILING_TERM],
        "worst_acceptable_grade_binding_rule_id": binder[CEILING_GRADE],
        "amount_cap_chain_rule_ids": chain_ids[CEILING_AMOUNT], "amount_cap_chain_values": chain_values[CEILING_AMOUNT],
        "term_cap_chain_rule_ids": chain_ids[CEILING_TERM], "term_cap_chain_values": chain_values[CEILING_TERM],
        "worst_acceptable_grade_chain_rule_ids": chain_ids[CEILING_GRADE],
        "worst_acceptable_grade_chain_values": chain_values[CEILING_GRADE],
        "waterfall_rule_ids": rule_ids, "waterfall_rule_owners": rule_owners, "waterfall_rule_ceilings": rule_ceilings,
        "waterfall_rule_status": rule_status, "waterfall_rule_value_before": rule_before,
        "waterfall_rule_value_after": rule_after,
        "waterfall_decline_reason_codes": decline_reasons,
        "uplift_authority_reference": uplift_authority_reference, "uplift_restrained_by_rule_id": uplift_restrained_by,
    }


_READS = ["risk_grade", "channel_code", "employment_type_code", "internal_tenure_months", "months_employed",
          "worst_arrears_months", "accounts_in_arrears_count", "bureau_enquiry_count_60d", "employer_id",
          "group_exposure_limit", "internal_exposure_total", "campaign_id", "campaign_is_preapproved",
          "purpose_code", "residency_code", "requested_term_months", "applicant_age_years"]
_WRITES = ["amount_cap", "term_cap", "worst_acceptable_grade", "amount_cap_binding_rule_id",
           "term_cap_binding_rule_id", "worst_acceptable_grade_binding_rule_id", "amount_cap_chain_rule_ids",
           "amount_cap_chain_values", "term_cap_chain_rule_ids", "term_cap_chain_values",
           "worst_acceptable_grade_chain_rule_ids", "worst_acceptable_grade_chain_values", "waterfall_rule_ids",
           "waterfall_rule_owners", "waterfall_rule_ceilings", "waterfall_rule_status",
           "waterfall_rule_value_before", "waterfall_rule_value_after", "waterfall_decline_reason_codes",
           "uplift_authority_reference", "uplift_restrained_by_rule_id"]


@frame_step(reads=_READS, writes=_WRITES)
def cap_waterfall(df):
    import polars as pl
    results = [_run_waterfall(row) for row in df.select(_READS).to_dicts()]
    return df.with_columns(pl.DataFrame(results))
