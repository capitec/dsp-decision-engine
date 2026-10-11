"""Every capability once, over the shipped evidence.

    uv run python scripts/tour.py
"""
import dataclasses
import tempfile
from datetime import date
from pathlib import Path

from governance import overlays
from governance.certify import certify
from governance.coverage import coverage
from governance.evidence import EvidenceStore
from governance.explain import explain
from governance.flows import BUSINESS, LOAN
from governance.replay import replay
from governance.swapset import attribute

store = EvidenceStore(Path(__file__).resolve().parents[1] / "evidence")
loan = store.load("loan", "d5f5f6b0-3f8a-4f2a-9c4b-2f6e7a1b0003")

print("replay:", replay(loan).verdict)
asked_for_less = dataclasses.replace(loan, request={**loan.request, "requested_amount": 1000.0})
print("replay, request edited:", [d.field for d in replay(asked_for_less).divergences][:4])

declined = store.load("business", "biz-2026-0900123")
for audience in ("consultant", "analyst"):
    print(f"explain {audience}:", explain(declined, audience).text)

print("overlays due on 2026-10-01:", [(r["flow"], r["adjustment_id"], r["age_days"]) for r in overlays.ageing(date(2026, 10, 1))])
print("unwind fraud:", overlays.unwind_estimate(store.all("fraud")))

population = [e.request for e in store.all("business")]
with tempfile.TemporaryDirectory() as tmp:
    def refer_from_6(config):
        config["params"]["business_credit"]["decision"]["outcome"]["business_outcome"]["refer_cutoff_grade"] = 6
    runs = [("base", BUSINESS.replayer()),
            ("overlays off", BUSINESS.replayer(params=overlays.switched_off(BUSINESS.replayer().module_fn().params))),
            ("refer from grade 6", BUSINESS.candidate(refer_from_6, Path(tmp)))]
    for s in attribute(BUSINESS, population, runs):
        print("swap set:", s.label, len(s.moved), "moved", s.reasons_appeared)

c = coverage(LOAN, [e.response for e in store.all("loan")])
print("loan rules never bound:", len(c.dead), "of", len(c.hits), "; dominant:", c.dominant)

def raise_grade_12(config):
    for row in config["rate_card_flex_loan"]["rows"]:
        if row["grade"] == 12:
            row["rate"] += 1.0

with tempfile.TemporaryDirectory() as tmp:
    worst = {e.decision_id for e in store.all("loan") if e.response["risk_grade"] == 12}
    cert = certify(store.all("loan"), LOAN.candidate(raise_grade_12, Path(tmp)), worst, ("CRC-2026-14",), "J. Moyo")
    print("certificate:", cert.passed, len(cert.moved), "moved,", cert.changed_documents)
