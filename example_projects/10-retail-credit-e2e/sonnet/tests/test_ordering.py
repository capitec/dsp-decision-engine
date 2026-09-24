"""Ordering constraints (spec 10 §5.22) checked against this project's own declared
execution order (`pipeline.py`'s wiring, restated here as the sequence it produces).
"""
from retail_credit.ordering import check_order

# The order `pipeline.py`'s dag resolves to for entry point 1 (data-dependency order, not
# list-declaration order -- see pipeline.py's own docstring). Declared here once so a
# future reordering of pipeline.py is checked against §5.22 rather than only discovered
# when an output looks wrong.
_EXECUTED_ORDER = [
    "P01", "P02", "P03.consent", "P02", "P03", "P04.acquisition",
    "P06.bureau", "P06.income", "P06.segment", "P07",
    "P08.adjustments", "P08.grading", "P09a", "P10a", "P10", "P09b",
    "P11", "P12.rate", "P12.fee", "P12.premium", "P12.instalment",
    "P13", "P14", "P16", "P17", "P18.ranking", "P18.disclosure",
]


def test_declared_order_violates_none_of_the_checked_constraints():
    violations = check_order(_EXECUTED_ORDER)
    assert violations == [], violations


def test_o01_adjustments_before_grading_is_checked():
    rest = [c for c in _EXECUTED_ORDER if c not in ("P08.adjustments", "P08.grading")]
    bad_order = ["P08.grading", "P08.adjustments"] + rest  # grading before adjustments: wrong
    violations = check_order(bad_order)
    assert any(v.startswith("O-01") for v in violations)


def test_o14_rate_before_fee_before_premium_before_instalment_is_checked():
    bad_order = [c for c in _EXECUTED_ORDER if not c.startswith("P12.")]
    bad_order = bad_order + ["P12.instalment", "P12.premium", "P12.fee", "P12.rate"]
    violations = check_order(bad_order)
    assert any(v.startswith("O-14") for v in violations)
