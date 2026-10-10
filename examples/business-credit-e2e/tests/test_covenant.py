"""The DSCR covenant: the version binds once, and a test keeps its three dates apart."""
import warnings
from datetime import date

import pytest

from decikit import HiddenStateWarning, InterpretedEngine, flow
from decikit.lab import diff

from business_credit_e2e import covenant, vocab
from business_credit_e2e.pipeline import build


def run_test(instance, ebitda=1_200_000.0, debt_service=80_000.0, test=date(2029, 6, 30), delivery=date(2029, 8, 29),
             determination=date(2029, 9, 4)):
    return covenant.evaluate(instance, ebitda=ebitda, monthly_debt_service=debt_service, test_date=test,
                             delivery_date=delivery, determination_date=determination)


def test_a_new_covenant_is_bound_to_the_version_in_force_at_approval():
    assert covenant.bind_covenant(1, date(2025, 6, 1))["covenant_definition_version"] == "dscr-2024.01"
    assert covenant.bind_covenant(2, date(2027, 1, 1))["covenant_definition_version"] == "dscr-2026.03"


def test_a_covenant_bound_in_2026_is_still_tested_on_that_version_in_2029():
    result = run_test(covenant.bind_covenant(3, date(2026, 6, 1)))
    assert result["covenant_definition_version"] == "dscr-2026.03"  # not dscr-2028.09, in force by the 2029 test
    assert covenant.load_covenant_documents()[0].version_on(result["test_date"]).version_id == "dscr-2028.09"


def test_three_dates_are_recorded_as_given():
    result = run_test(covenant.bind_covenant(4, date(2026, 1, 1)))
    assert (result["test_date"], result["delivery_date"], result["determination_date"]) == (
        date(2029, 6, 30), date(2029, 8, 29), date(2029, 9, 4))


@pytest.mark.parametrize("debt_service, breach", [
    (80_000.0, vocab.BREACH_NONE),       # 1.25 exactly meets the threshold
    (88_000.0, vocab.BREACH_TECHNICAL),  # 1.136: within 10% of it
    (100_000.0, vocab.BREACH_MATERIAL),  # 1.0: within 25%
    (120_000.0, vocab.BREACH_SEVERE),    # 0.833
])
def test_a_breach_is_classed_by_how_far_the_ratio_falls_short(debt_service, breach):
    instance = {**covenant.bind_covenant(5, date(2026, 6, 1))}
    assert run_test(instance, ebitda=1_200_000.0, debt_service=debt_service)["breach_class_code"] == breach


def test_the_cure_clock_runs_from_the_earlier_of_delivery_and_determination():
    instance = covenant.bind_covenant(6, date(2026, 6, 1))
    late_decision = run_test(instance, debt_service=100_000.0, delivery=date(2029, 8, 29), determination=date(2029, 9, 4))
    assert late_decision["cure_days_remaining"] == 30 - 6
    on_arrival = run_test(instance, debt_service=100_000.0, delivery=date(2029, 8, 29), determination=date(2029, 8, 29))
    assert on_arrival["cure_days_remaining"] == 30
    assert run_test(instance)["cure_days_remaining"] is None  # no breach, nothing to cure


def test_a_missing_certificate_is_not_tested_and_not_a_financial_result():
    result = covenant.certificate_not_received(covenant.bind_covenant(7, date(2026, 1, 1)), date(2027, 6, 30), date(2027, 9, 1))
    assert result["breach_class_code"] == vocab.COVENANT_NOT_TESTED
    assert result["measured_dscr"] is None and result["information_undertaking_breached"] is True


def test_no_debt_service_is_an_infinite_ratio_and_no_breach():
    assert run_test(covenant.bind_covenant(8, date(2026, 6, 1)), debt_service=0.0)["breach_class_code"] == vocab.BREACH_NONE


def test_a_changed_threshold_in_the_terms_document_is_a_different_pipeline():
    versions, terms = covenant.load_covenant_documents()
    stricter = terms.model_copy(update={"terms": tuple(
        t.model_copy(update={"threshold": 1.5}) if t.version_id == "dscr-2026.03" else t for t in terms.terms)})
    covenant_flow = lambda v, t: flow(v, t, name="covenant")  # noqa: E731
    assert diff.structure(covenant_flow(versions, terms), covenant_flow(*covenant.load_covenant_documents())).identical
    assert not diff.structure(covenant_flow(versions, terms), covenant_flow(versions, stricter)).identical
    assert stricter.for_version("dscr-2026.03") == (1.5, 30)


def test_the_covenant_steps_read_no_hidden_state():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InterpretedEngine().bind(build())
    assert [str(w.message) for w in caught if issubclass(w.category, HiddenStateWarning)
            and ("covenant" in str(w.message) or "COV-DSCR" in str(w.message))] == []
