"""Every example, built from its latest config version, reads no hidden Python object and shadows no seeded table."""
import warnings
from pathlib import Path

import pytest

from decider import Engine, HiddenStateWarning, ShadowedTableWarning
from decider.config import JsonFileStore
from decider.serving import RequestHandler

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


@pytest.mark.parametrize("project", sorted(PIPELINES))
def test_an_example_reads_no_hidden_state_and_shadows_no_table(project, monkeypatch):
    for path in EXAMPLES.iterdir():
        if path.is_dir():
            monkeypatch.syspath_prepend(str(path))
    store = JsonFileStore(basepath=str(EXAMPLES / project / "configs"))
    version = store.read(store.latest_version())
    handler = RequestHandler(store, f"{PIPELINES[project]}.pipeline:build")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exe = Engine().bind(handler.pipeline_fn(version.config), mode="interpreted")
        exe.warn_shadowed_tables(version.config.get("params"))
    found = [str(w.message) for w in caught if issubclass(w.category, (HiddenStateWarning, ShadowedTableWarning))]
    unexpected = [m for m in found if not any(allowed in m for allowed in ALLOWED.get(project, ()))]
    assert unexpected == []
    assert (len(found) > 0) == (project in ALLOWED)  # an allow-list entry nothing needs any more is removed
