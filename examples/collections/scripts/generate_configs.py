"""Write `configs/<version>/`: the treatment matrix, the six overlay stacks and the params document.

    uv run python examples/collections/scripts/generate_configs.py [version]
"""
import json
import sys
from pathlib import Path

from decider import ConfigurableStep, DecisionTableConfig

from collections_treatment import capacity, matrix
from collections_treatment import vocab as v
from collections_treatment.pipeline import build


def overlay(adjustment_id, kind, target, op, value, scope, stack_position, owner, approval_reference, rationale,
            effective_from, effective_to, review_date, tighten_only=False):
    return {"adjustment_id": adjustment_id, "kind": kind, "target": target, "effect": {"op": op, "value": value},
            "scope": scope, "stack_position": stack_position, "owner": owner, "approval_reference": approval_reference,
            "rationale": rationale, "effective_from": effective_from, "effective_to": effective_to,
            "review_date": review_date, "tighten_only": tighten_only}


def stack(name, target, set_id, base_field, adjusted_output, adjustments, scope_types=None):
    return {"type": "adjustment_stack", "name": name, "target": target, "set_id": set_id,
            "base_field": base_field, "adjusted_output": adjusted_output, "adjustments": adjustments,
            **({"scope_types": scope_types} if scope_types else {})}


STACKS = [
    stack("score_shift", "collections_score", "AS-COLL-SCORE-2026.09", "collections_score_unadjusted",
          "collections_score", [
        overlay("ADJ-COLL-SCORE-001", "score_shift", "collections_score", "add", -22.0, {"agency_placed_12m": True}, 1,
                "Collections Strategy", "MRV-2026-014",
                "Accounts placed with an agency in the last 12 months score optimistic on the base curve",
                "2026-01-01", "2026-12-31", "2026-10-01", tighten_only=True)], {"agency_placed_12m": "bool"}),
    stack("scaling_change", "calibration_scale", "AS-COLL-SCORE-2026.09", "scale", "adjusted_scale", [
        overlay("ADJ-COLL-SCORE-003", "scaling_change", "calibration_scale", "multiply", 23.0 / 20.0,
                {"product_family_code": 2}, 1, "Model team", "MRV-2026-021",
                "Points-to-double-the-odds moved from 20 to 23 for the revolving segment",
                "2026-03-01", "2027-01-01", "2026-12-01")]),
    stack("odds_multiplier", "roll_probability", "AS-COLL-SCORE-2026.09", "roll_probability_before_overlay",
          "roll_probability_overlaid", [
        overlay("ADJ-COLL-SCORE-002", "odds_multiplier", "roll_probability", "multiply", 1.28,
                {"product_family_code": 1}, 2, "Model Risk and Validation", "MRV-2026-019",
                "Deteriorating quarter on the unsecured book", "2026-06-01", "2027-03-01", "2026-12-01",
                tighten_only=True)]),
    stack("band_boundary_shift", "collections_band_boundary", "AS-COLL-SCORE-2026.09", "band_edge_origin",
          "band_edge_shift", [
        overlay("ADJ-COLL-SCORE-004", "band_boundary_shift", "collections_band_boundary", "add", 0.03, {}, 1,
                "Collections Strategy", "CCF-2026-028", "Move the band 3/4 edge up by 0.03 of roll probability",
                "2026-05-01", "2026-11-01", "2026-10-01")]),
    stack("intensity_dial", "matrix_treatment_intensity", "AS-COLL-MATRIX-2026.09", "matrix_treatment_intensity",
          "dialled_intensity", [
        overlay("ADJ-COLL-MATRIX-001", "cap_adjustment", "matrix_treatment_intensity", "add", 1.0,
                {"arrears_bucket_code": 3}, 1, "Collections Strategy", "CCF-2026-033",
                "Escalate bucket 3 by one intensity level for six weeks", "2026-08-01", "2026-09-12", "2026-09-05")]),
    stack("treatment_suppression", "matrix_treatment_code", "AS-COLL-MATRIX-2026.09", "matrix_treatment_code",
          "suppressed_treatment_code", [
        overlay("ADJ-COLL-MATRIX-002", "treatment_suppression", "matrix_treatment_code", "set", float(v.NO_ACTION),
                {"matrix_treatment_code": v.FIELD_VISIT}, 2, "Collections Strategy", "CCF-2026-041",
                "Suspend field visits for the quarter", "2026-07-01", "2026-10-01", "2026-09-15")]),
]


def main(version: str = "0.1.0") -> None:
    out = Path(__file__).resolve().parents[1] / "configs" / version
    out.mkdir(parents=True, exist_ok=True)
    document = matrix.generate_matrix()
    (out / "treatment_matrix.json").write_text(json.dumps(document, separators=(",", ":")))
    for doc in STACKS:
        (out / f"{doc['name']}.json").write_text(json.dumps(doc, indent=2) + "\n")
    stacks = [ConfigurableStep.load(doc) for doc in STACKS]
    params = build(DecisionTableConfig.load(document), *stacks).parameters().defaults()
    params["collections_treatment"]["capacity"]["pools"] = capacity.default_pool_rows()
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {len(document['rows'])} matrix cells, {len(STACKS)} overlay stacks and {out / 'params.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
