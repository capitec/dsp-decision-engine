"""BANDS -> a JDM document (inputNode -> decisionTableNode -> outputNode),
the same shape as the ZEN repo's own test-data/table.json and
credit-analysis.json's "Turnover" node.
"""
from __future__ import annotations

from rules import BANDS


def _cell(lower, upper) -> str:
    """A half-open range as a ZEN unary test — `[a..b)`, `< b` or `>= a`
    for the open edges, matching BoundMode.lower_inclusive exactly (probed
    directly against zen.evaluate_unary_expression before use)."""
    if lower is None and upper is None:
        return ""  # wildcard — not used by BANDS, included for completeness
    if lower is None:
        return f"< {upper}"
    if upper is None:
        return f">= {lower}"
    return f"[{lower}..{upper})"


def build_content() -> dict:
    rules = []
    for i, (lo, hi, code, name, limit) in enumerate(BANDS):
        rules.append({
            "_id": f"row{i}",
            "income": _cell(lo, hi),
            "tier_code": str(code),
            "tier_name": f"\"{name}\"",
            "limit": str(limit),
        })

    input_id = "3e3f5093-c969-4c3a-97e1-560e4b769a12"
    table_id = "0624d5fd-1944-4781-92bb-e32873ce91e2"
    output_id = "e0438c6b-dee0-405e-a941-9b4c3d9c4b83"

    return {
        "nodes": [
            {"id": input_id, "type": "inputNode", "name": "Request", "position": {"x": 0, "y": 0}},
            {
                "id": table_id,
                "type": "decisionTableNode",
                "name": "IncomeBand",
                "position": {"x": 200, "y": 0},
                "content": {
                    "hitPolicy": "first",
                    "inputs": [
                        {"id": "income", "name": "Income", "field": "income", "type": "expression"},
                    ],
                    "outputs": [
                        {"id": "tier_code", "name": "Tier code", "field": "tier_code", "type": "expression"},
                        {"id": "tier_name", "name": "Tier name", "field": "tier_name", "type": "expression"},
                        {"id": "limit", "name": "Limit", "field": "limit", "type": "expression"},
                    ],
                    "rules": rules,
                },
            },
            {"id": output_id, "type": "outputNode", "name": "Response", "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "sourceId": input_id, "targetId": table_id, "type": "edge"},
            {"id": "e2", "sourceId": table_id, "targetId": output_id, "type": "edge"},
        ],
    }
