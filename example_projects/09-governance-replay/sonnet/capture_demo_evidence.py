"""Populates `evidence_store/` with decisions this harness can replay, explain,
swap-set and measure coverage over.

None of 01/03/05 persist evidence externally (see `governance/evidence_store.py`'s
own docstring for why -- their `.score()` output already satisfies 09 §5.15's
per-decision contract; only durable storage is missing, and that is this
harness's concern, not theirs). This script plays that missing "write the
evidence somewhere" step for each flow's own `sample_request.json`, plus a
small synthetic population per flow (a deterministic sweep over one or two
numeric fields, not a random sample -- 09 §5.15 item 3's own "no RNG, ever"
discipline applies just as much to generating demo data as to a flow's own
logic) so the coverage, ageing and swap-set capabilities have more than one
decision to work over.

Run with: `uv run --project <REPO> python capture_demo_evidence.py`
(PYTHONPATH must already include 00, 01, 02, 03 and 05 -- see SERVE.md).
"""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from governance import evidence_store, flows

ROOT = Path(__file__).resolve().parent


def _load_sample(flow_code: str) -> dict:
    project_dir = flows.get(flow_code).project_dir()
    return json.loads((project_dir / "sample_request.json").read_text())


def _variant(base: dict, decision_id: str, **overrides) -> dict:
    record = copy.deepcopy(base)
    record.update(overrides)
    record["decision_id"] = decision_id
    return record


def _capture_all(flow_code: str, requests: list[dict], config_version: str = "0.1.0") -> None:
    adapter = flows.get(flow_code)
    built = adapter.build(config_version)
    for request in requests:
        evidence = evidence_store.capture(adapter, built, request)
        path = evidence_store.save(evidence)
        print(f"  {flow_code} {evidence.decision_id} -> {path}")


def fraud_population() -> list[dict]:
    base = _load_sample("01")
    requests = [base]
    # A deterministic sweep over amount and channel -- enough spread that some rules
    # fire on every record (dominant) and some never do (dead), which is the point.
    for i, (amount, channel, device_hours) in enumerate([
        (500.0, 1, 400.0), (2000.0, 2, 200.0), (4500.0, 4, 48.0), (9200.0, 4, 12.0),
        (15000.0, 4, 1.0), (25000.0, 3, 0.2), (60.0, 1, 720.0), (150000.0, 4, 0.05),
    ], start=1):
        requests.append(_variant(base, f"9f2b6f0e-demo-{i:04d}", amount=amount, channel_code=channel,
                                  device_change_hours=device_hours))
    return requests


def granting_population() -> list[dict]:
    base = _load_sample("03")
    requests = [base]
    for i, (amount, term, age) in enumerate([
        (30000.0, 12, 22), (50000.0, 24, 35), (80000.0, 36, 45), (120000.0, 48, 58),
        (15000.0, 6, 61), (95000.0, 60, 29), (45000.0, 18, 40), (70000.0, 30, 52),
    ], start=1):
        requests.append(_variant(base, f"gr-2026-demo-{i:04d}", requested_amount=amount,
                                  requested_term_months=term, applicant_age_years=age))
    return requests


def business_population() -> list[dict]:
    base = _load_sample("05")
    requests = [base]
    for i, amount in enumerate([250000.0, 500000.0, 900000.0, 1500000.0, 3200000.0, 180000.0, 4500000.0], start=1):
        requests.append(_variant(base, f"biz-2026-demo-{i:04d}", requested_amount=amount))
    return requests


def main() -> None:
    print("Capturing 01 (transaction fraud)...")
    _capture_all("01", fraud_population())
    print("Capturing 03 (unsecured granting and pricing)...")
    _capture_all("03", granting_population())
    print("Capturing 05 (business nested entities)...")
    _capture_all("05", business_population())


if __name__ == "__main__":
    main()
