import copy
import json
from datetime import date
from pathlib import Path

import pytest

from decikit import flow

from retail_credit import pricing, solve
from retail_credit.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    record["decision_date"] = date.fromisoformat(record["decision_date"])
    for account in record["bureau_accounts"] + record["internal_accounts"]:
        account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline():
    return build().compile("interpreted")


@pytest.fixture
def sample():
    return json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture
def decide(pipeline, sample, params):
    """`decide(**changes)` scores the sample request with the changes applied; `None` removes a field."""

    def run(base=None, *, params_document=None, **changes):
        record = copy.deepcopy(base or sample)
        for field, value in changes.items():
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
        return pipeline.run(parse_dates(record), params_document or params)

    return run


@pytest.fixture(scope="session")
def solve_once(params):
    """`solve_once(**changes)` runs the pricing and solve steps alone on a request with an explicit instalment ceiling."""
    unit = flow(pricing.amount_independent_unit(), solve.solve_unit(), name="solve_only")
    pipeline_params = params["retail_credit"]  # the same policy numbers and card as the whole pipeline
    unit_params = {"solve_only": {"solve_offer": pipeline_params["solve_offer"], "fixed_terms": pipeline_params["fixed_terms"]}}
    executable = unit.compile("interpreted")
    request = {"requested_amount": 95_000.0, "amount_cap": 95_000.0, "product_min_amount": 2_000.0,
               "product_max_amount": 500_000.0, "amount_grid": 250.0, "risk_grade": 7, "applicant_age_years": 34.0, "cover_type_code": 1,
               "offer_term_months": 60.0, "rate_addon_bps": 0.0, "instalment_ceiling": 2_500.0}
    return lambda **changes: executable.run({**request, **changes}, unit_params)


