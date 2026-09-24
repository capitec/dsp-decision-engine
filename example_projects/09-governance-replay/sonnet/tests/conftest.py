import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve().parents[1]  # this project's own root (contains pipeline.py)

_SIBLINGS = (
    "00-shared-credit-core", "01-transaction-fraud", "02-affordability",
    "03-loan-granting-pricing", "05-business-nested",
)


def _find_sibling(name: str) -> Path:
    """Same two-layout resolution 01/03/05's own `tests/conftest.py` uses for
    `00-shared-credit-core`, generalised to every flow this harness consumes."""
    candidates = [
        _THIS.parent / name,                          # scratch: direct sibling
        _THIS.parents[1] / name / _THIS.name,          # repo: example_projects/<name>/<model>
    ]
    for candidate in candidates:
        if (candidate / "pipeline.py").is_file():
            return candidate
    raise FileNotFoundError(f"could not find sibling project {name!r} near {_THIS}; tried {candidates}")


# Every consumed flow's root goes on `sys.path` first (for `credit_core`, `fraud_interdiction`,
# `assessment`, `loan_granting`, `business_nested`), then THIS project's own root last, so it
# ends up first -- the same ordering discipline 01/03/05's own conftest.py documents, needed
# because every one of these projects ships a `pipeline.py` at its root.
for _name in _SIBLINGS:
    sys.path.insert(0, str(_find_sibling(_name)))
sys.path.insert(0, str(_THIS))


@pytest.fixture(autouse=True, scope="session")
def _isolated_evidence_store(tmp_path_factory):
    """Redirects `governance.evidence_store.STORE_ROOT` to a temporary directory for the
    whole test session, so tests never depend on, or pollute, this project's own checked-in
    `evidence_store/` (populated separately by `capture_demo_evidence.py`, per SERVE.md)."""
    from governance import evidence_store

    original = evidence_store.STORE_ROOT
    evidence_store.STORE_ROOT = tmp_path_factory.mktemp("evidence_store")
    yield
    evidence_store.STORE_ROOT = original
