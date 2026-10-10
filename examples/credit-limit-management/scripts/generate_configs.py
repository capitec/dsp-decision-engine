"""Write `configs/<version>/`: the 1 152-cell limit matrix, the overlay and reason-code documents, and the params document.

    uv run python examples/credit-limit-management/scripts/generate_configs.py [version]

The params document is every default, then the programme's own policy: the income haircuts for deposit evidence
(none on a steady salary, 15% on irregular deposits, none on declared income, which is indexed for inflation
instead) and how old a bureau view may be before the assessment cannot use it at all. Automatic against
conditional is decided by the programme's own, tighter 35 days.

The overlays are `adjustment_stack` documents, one per value they overlay, and the reason codes a `reason_codes`
document; a business user edits those files, and `pipeline.build` takes them by name.
"""
import json
import sys
from pathlib import Path

from decikit import ConfigurableStep

from limit_mgmt import reasons as r
from limit_mgmt.matrix import build_matrix_table, matrix_document
from limit_mgmt.pipeline import build

SET_ID = "AS-07-2026.09"


def overlay(adjustment_id, kind, target, op, value, scope, owner, approval, rationale, effective_from, effective_to,
            review_date):
    return {"adjustment_id": adjustment_id, "kind": kind, "target": target, "effect": {"op": op, "value": value},
            "scope": scope, "stack_position": 1, "owner": owner, "approval_reference": approval, "rationale": rationale,
            "effective_from": effective_from, "effective_to": effective_to, "review_date": review_date,
            "tighten_only": True}


def stack(target, overlays, **fields):
    return {"type": "adjustment_stack", "name": f"apply_{target}_adjustments", "target": target, "set_id": SET_ID,
            **fields, "adjustments": overlays}


# The programme underwrites against a salary pattern and a bureau file, not evidence collected days earlier, so its
# affordability buffer is wider than origination's. It is an overlay on the shared buffer grid, not a fork of it.
DOCUMENTS = {
    "behaviour_score_adjustments": stack("behaviour_score", [overlay(
        "ADJ-07-001", "score_shift", "behaviour_score", "add", -15.0, {"channel_code": 5}, "Model Risk", "MRC-2026-041",
        "Partner-originated channel running worse than model prediction in early life",
        "2026-01-01", None, "2026-12-31")], base_field="behaviour_score_raw",
        unadjusted_output="behaviour_score_unadjusted"),
    "behaviour_pd_adjustments": stack("probability_of_default", [overlay(
        "ADJ-07-002", "odds_multiplier", "probability_of_default", "multiply", 1.25, {"product_code": 21}, "Model Risk",
        "MRC-2026-044", "Access Facility observed default running above prediction",
        "2026-03-01", "2027-03-01", "2026-12-01")], base_field="probability_of_default_raw",
        unadjusted_output="probability_of_default_unadjusted"),
    "matrix_dial_adjustments": stack("matrix_multiplier_excess", [overlay(
        "ADJ-07-DIAL-2026-03", "cap_adjustment", "matrix_multiplier_excess", "multiply", 0.80, {}, "Credit Risk Policy",
        "CRC-2026-088", "Funding environment tightened; run the programme at 80% of the matrix this cycle",
        "2026-09-01", "2026-11-01", "2026-10-15")], adjusted_output="matrix_multiplier_excess_adjusted",
        unadjusted_output="matrix_multiplier_excess_unadjusted"),
    "buffer_adjustments": stack("affordability_buffer_applied", [overlay(
        "ADJ-07-BUFFER-BASE", "buffer_adjustment", "affordability_buffer_applied", "add", 0.06, {},
        "Credit Risk Policy", "CRC-2026-091", "Programme affordability rests on inferred income and a monthly bureau file",
        "2026-01-01", None, "2026-12-31")], base_field="affordability_buffer_applied",
        unadjusted_output="affordability_buffer_unadjusted"),
    "reason_codes": {"type": "reason_codes", "name": "rank_reasons", "version": "limit-mgmt-reasons-2026.09", "codes": [
        {"code": code, "severity_rank": rank, "description": text, "is_regulatory": regulatory} for code, rank, text, regulatory in (
            (r.R_EXCLUDED, 5, "Account excluded from the automatic programme", True),
            (r.R_AFFORDABILITY_FAIL, 10, "The client cannot carry any of the increase", True),
            (r.R_AFFORDABILITY_UNESTABLISHED, 12, "Affordability could not be established on the evidence available", True),
            (r.R_SUPPRESSED_BY_CLIENT_DECREASE, 15,
             "Suppressed: an immediate decrease fired on another account of this client", True),
            (r.R_REDUCED_BELOW_MINIMUM_BY_OVERLAY, 25, "Reduced below the minimum meaningful increase by an active overlay", False),
            (r.R_NO_INCREASE_AFTER_CAPS, 30,
             "The matrix and the caps leave no increase, or one under the cell's minimum increment", False),
            (r.R_BELOW_MINIMUM_MEANINGFUL_INCREASE, 35, "Increase below the minimum meaningful amount", False),
            (r.R_BELOW_FUNDING_LINE, 40, "Ranked below the funding line this cycle", False),
            (r.R_TAIL_SKIPPED, 42, "Did not fit the remaining budget; skipped", False),
            (r.R_FAIRNESS_CAPPED, 45, "Segment fairness floor reached", False))]},
}

_INCOME = "affordability_bridge/evidence/applicant{n}_income/income_haircut_applied"
POLICY = {
    **{f"{_INCOME.format(n=n)}/{name}": value for n in (1, 2) for name, value in (
        ("haircut_statement_high_confidence", 0.0), ("haircut_statement_low_confidence", 0.15), ("haircut_declared", 0.0))},
    "affordability_bridge/evidence/bureau_is_stale/staleness_window_days": 60,
}


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    params = build(build_matrix_table(), **{name: ConfigurableStep.load(doc) for name, doc in DOCUMENTS.items()}
                   ).parameters().defaults()
    for path, value in POLICY.items():
        *keys, name = ["limit_mgmt", *path.split("/")]
        node = params
        for key in keys:
            node = node[key]
        assert name in node, path
        node[name] = value
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "matrix.json").write_text(json.dumps(matrix_document(), indent=2) + "\n")
    for name, doc in DOCUMENTS.items():
        (out / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}: params.json, matrix.json and {', '.join(f'{name}.json' for name in DOCUMENTS)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
