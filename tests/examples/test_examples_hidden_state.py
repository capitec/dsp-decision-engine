"""Every example, built from its latest config version, reads no hidden Python object and shadows no seeded table."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PIPELINES = {
    "affordability": "assessment", "business-credit-e2e": "business_credit_e2e", "business-nested": "business_nested",
    "campaign-trees": "campaign_trees", "collections": "collections_treatment", "consolidation": "consolidation",
    "credit-limit-management": "limit_mgmt", "governance-replay": "governance", "loan-granting-pricing": "loan_granting",
    "retail-credit-e2e": "retail_credit", "shared-credit-core": "credit_core", "transaction-fraud": "fraud_interdiction",
}
# The replayer's registry of the three decision flows it replays is its declared boundary: it is the thing the
# harness exists to hold, and it is not a step of any one flow.
ALLOWED = {"governance-replay": ("`FLOWS`",)}


def test_every_example_is_listed():
    assert sorted(PIPELINES) == sorted(p.name for p in EXAMPLES.iterdir() if (p / "README.md").is_file())


# One process per project: the example packages register config types that other tests in this process also claim.
SCRIPT = """
import json, sys, warnings
from decikit import Engine, HiddenStateWarning, ShadowedTableWarning
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

project, package, configs = sys.argv[1:]
store = JsonFileStore(basepath=configs)
version = store.read(store.latest_version())
handler = RequestHandler(store, package + ".pipeline:build")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    exe = Engine().bind(handler.pipeline_fn(version.config), mode="interpreted")
    exe.warn_shadowed_tables(version.config.get("params"))
print(json.dumps([str(w.message) for w in caught if issubclass(w.category, (HiddenStateWarning, ShadowedTableWarning))]))
"""


@pytest.mark.parametrize("project", sorted(PIPELINES))
def test_an_example_reads_no_hidden_state_and_shadows_no_table(project):
    path = os.pathsep.join(str(p) for p in sorted(EXAMPLES.iterdir()) if p.is_dir())
    run = subprocess.run([sys.executable, "-c", SCRIPT, project, PIPELINES[project], str(EXAMPLES / project / "configs")],
                         env={**os.environ, "PYTHONPATH": path}, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    found = json.loads(run.stdout.splitlines()[-1])
    unexpected = [m for m in found if not any(allowed in m for allowed in ALLOWED.get(project, ()))]
    assert unexpected == []
    assert (len(found) > 0) == (project in ALLOWED)  # an allow-list entry nothing needs any more is removed
