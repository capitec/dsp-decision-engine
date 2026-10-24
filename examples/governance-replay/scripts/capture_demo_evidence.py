"""Capture the demo evidence: the sample request of each flow and variants that decide differently.

    uv run python scripts/capture_demo_evidence.py
"""
import copy
import json
from pathlib import Path

from governance.evidence import EvidenceStore, capture
from governance.flows import BUSINESS, FRAUD, LOAN

ROOT = Path(__file__).resolve().parents[1]

# A holder of 15% with six judgments, two of them disputed: a referral, not a decline.
HOLDER = {"entity_id": 2, "entity_key": "HOLDER", "is_natural_person": True, "relationship_type_code": 5,
          "effective_ownership_pct": 15, "bureau_score": 705, "months_on_record": 100, "worst_delinquency_months": 0,
          "adverse_events": [{"event_id": i, "event_type_code": 1, "amount": 90000, "event_date": f"2024-0{i}-01",
                              "is_disputed": i <= 2} for i in range(1, 7)]}
OWNER = {"entity_id": 1, "entity_key": "OWNER", "is_natural_person": True, "relationship_type_code": 5,
         "effective_ownership_pct": 85, "bureau_score": 705, "months_on_record": 100, "worst_delinquency_months": 0,
         "adverse_events": []}


def edits(base: dict, *changes: dict) -> list[dict]:
    out = [base]
    for i, change in enumerate(changes, 1):
        request = {**copy.deepcopy(base), **change}
        for gone in [k for k, v in request.items() if v is None]:
            del request[gone]
        request["decision_id"] = f"{base['decision_id']}-v{i}"
        out.append(request)
    return out


def populations() -> dict:
    sample = lambda flow: json.loads((flow.project() / "sample_request.json").read_text())  # noqa: E731
    business = sample(BUSINESS)
    spotless = [{**e, "adverse_events": [v for v in e["adverse_events"] if v["event_id"] != 41]}
                for e in business["entities"]]
    return {
        FRAUD: edits(sample(FRAUD), {"amount": 300.0}, {"amount": 6000.0}, {"sanctions_list_match": True},
                     {"account_frozen": True, "amount": 50.0}, {"amount": 150000.0},
                     {"decision_date": "2026-03-12"}, {"decision_date": "2026-02-01", "amount": 6000.0},
                     {"device_blocked": True}),
        LOAN: edits(sample(LOAN), {"requested_amount": None, "requested_term_months": None},
                    {"channel_code": 4, "bureau_score": 480, "revolving_utilisation": 0.8},
                    {"requested_amount": 30000.0, "requested_term_months": 12},
                    {"requested_amount": 500000.0, "requested_term_months": 84}, {"applicant_age_years": 17.0},
                    {"campaign_is_preapproved": False}, {"bureau_score": 600.0, "worst_arrears_months": 5.0}),
        BUSINESS: edits(business, {"entities": spotless}, {"entities": spotless, "requested_amount": 1000000.0},
                        {"requested_amount": 250000.0}, {"requested_amount": 800000.0, "entities": [OWNER, HOLDER]}),
    }


if __name__ == "__main__":
    store = EvidenceStore(ROOT / "evidence")
    for flow, requests in populations().items():
        for request in requests:
            e = capture(flow, request)
            store.save(e)
            print(flow.code, e.decision_id, e.response[flow.outcome], e.response[flow.primary])
