"""What this example really reuses, measured: the example packages and `credit_core` modules a decision imports."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

EXAMPLES = ("credit_core", "business_nested", "assessment", "limit_mgmt", "consolidation")
PROBE = textwrap.dedent("""
    import json, sys
    from business_credit_e2e.pipeline import build
    from business_credit_e2e.records import parse_dates
    import business_credit_e2e.review, business_credit_e2e.covenant, business_credit_e2e.history
    request = parse_dates(json.load(open(sys.argv[1])))
    build().compile("interpreted").score(request, build().parameters().defaults())
    print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in %r)))
""" % (EXAMPLES,))


def test_a_decision_runs_on_the_assessment_and_the_core_and_never_touches_consolidation_or_the_limit_programme():
    sample = Path(__file__).resolve().parents[1] / "sample_request.json"
    out = subprocess.run([sys.executable, "-c", PROBE, str(sample)], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}).stdout
    loaded = json.loads(out)
    assert {m.split(".")[0] for m in loaded} == {"credit_core", "business_nested", "assessment"}
    assert {m.split(".")[1] for m in loaded if m.startswith("credit_core.")} == {
        "adjustments", "adverse_events", "affordability", "calibration", "dates", "deductions", "evidence",
        "expense_norms", "fees", "income", "instalment", "obligations", "reason_codes", "rounding", "vocab"}
