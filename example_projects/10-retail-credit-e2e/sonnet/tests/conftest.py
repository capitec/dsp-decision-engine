import datetime
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where project 00 lives differs between a scratch build (sibling directory,
# `<scratch>/<model>/00-shared-credit-core`) and this repo's own layout
# (`example_projects/00-shared-credit-core/<model>`) -- try both rather than hardcoding one,
# so this conftest works whether it is run from scratch or from its final copy in the repo.
_CANDIDATES = [
    PROJECT_ROOT.parent / "00-shared-credit-core",                       # scratch: sibling
    PROJECT_ROOT.parent.parent / "00-shared-credit-core" / PROJECT_ROOT.name,  # repo: example_projects/00-.../<model>
]
CREDIT_CORE_ROOT = next((p for p in _CANDIDATES if (p / "credit_core").is_dir()), _CANDIDATES[0])

for p in (str(CREDIT_CORE_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)  # PROJECT_ROOT inserted last -> sys.path[0]: see SERVE.md "PYTHONPATH order"


@pytest.fixture()
def sample_record() -> dict:
    record = json.loads((PROJECT_ROOT / "sample_request.json").read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    for account in record["bureau_accounts"] + record["internal_accounts"]:
        account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    return record


@pytest.fixture()
def engine():
    from decider.engine.run.engine import Engine
    from pipeline import build

    pl = build()
    exe = Engine().bind(pl, mode="interpreted")
    return exe


@pytest.fixture()
def params() -> dict:
    return json.loads((PROJECT_ROOT / "configs" / "0.1.0" / "params.json").read_text())
