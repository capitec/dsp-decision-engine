from pathlib import Path

import pytest

from governance.evidence import EvidenceStore
from governance.flows import BUSINESS, FLOWS, FRAUD, LOAN

ROOT = Path(__file__).resolve().parents[1]
BASE_IDS = {"fraud": "9f2b6f0e-2a5b-4b8a-9b0c-111111111111", "loan": "d5f5f6b0-3f8a-4f2a-9c4b-2f6e7a1b0003",
            "business": "biz-2026-0900123"}


@pytest.fixture(scope="session")
def evidence():
    return EvidenceStore(ROOT / "evidence")


@pytest.fixture(params=sorted(FLOWS))
def flow_code(request):
    return request.param


@pytest.fixture
def decision(evidence, flow_code):
    """The shipped sample decision of each flow."""
    return evidence.load(flow_code, BASE_IDS[flow_code])


def requests_of(evidence, flow):
    return [e.request for e in evidence.all(flow.code)]
