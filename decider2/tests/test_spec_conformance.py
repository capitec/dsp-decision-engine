"""Spec conformance beyond the flagship.

tests/test_flagship.py is doc 03 §1.1's example: three float-only bare
functions in a straight line with distinct names. Everything here is a shape
the spec requires and that test never exercises. Each test cites the sentence
it enforces.
"""
import numpy as np
import polars as pl
import pytest

from decider2 import flow, module, param, step


# ---------------------------------------------------------------------------
# B — the driver is built once per generation, and probing never runs a body
# ---------------------------------------------------------------------------

def test_a_probe_never_executes_the_authors_step_body():
    """Doc 05 §6: a genuine runtime bug 'must propagate identically in both
    compiled and fallback paths'. Probing with synthetic values raises errors
    on data the caller never supplied."""
    def inv(x: float) -> float:
        """Reciprocal about 1.0."""
        return 1.0 / (x - 1.0)

    p = flow(inv)
    frame = pl.DataFrame({"x": [5.0, 3.0]})
    assert p.apply(frame, mode="interpreted")["inv"].to_list() == [0.25, 0.5]
    assert p.apply(frame, mode="fused")["inv"].to_list() == [0.25, 0.5]


def test_repeated_calls_do_not_rebuild_the_driver():
    """Doc 02 §3.4: 'Compilation happens at image build, not at startup.'
    Doc 05 §6: 'Cache the decision per node so a doomed compile isn't retried
    every call.'"""
    import numba

    def a(x: float, y: float) -> float:
        """A."""
        return x - y

    p = flow(a)
    frame = pl.DataFrame({"x": [1.0] * 64, "y": [2.0] * 64})
    p.apply(frame, mode="fused")          # warm

    events = {"n": 0}
    original = numba.core.dispatcher.Dispatcher.compile

    def counting(self, sig):
        events["n"] += 1
        return original(self, sig)

    numba.core.dispatcher.Dispatcher.compile = counting
    try:
        for _ in range(3):
            p.apply(frame, mode="fused")
    finally:
        numba.core.dispatcher.Dispatcher.compile = original

    assert events["n"] == 0, f"{events['n']} compiles across 3 warm calls"


# ---------------------------------------------------------------------------
# A — the waterfall: a rule that narrows a value it also reads
# ---------------------------------------------------------------------------

def _waterfall():
    @step(output="term_cap")
    def product_ceiling(requested_term: float,
                        ceiling: float = param(60.0, ge=6, le=84)) -> float:
        """TERM-0040 — the product ceiling."""
        return min(requested_term, ceiling)

    @step(output="term_cap")
    def cap_by_income(term_cap: float, min_net_salary: float,
                      cap: float = param(48.0, ge=6, le=60),
                      floor: float = param(5000.0, ge=0)) -> float:
        """TERM-0042 — cap below the income floor."""
        return min(term_cap, cap) if min_net_salary < floor else term_cap

    return product_ceiling, cap_by_income


WATERFALL_FRAME = pl.DataFrame({
    "requested_term": [72.0, 72.0],
    "min_net_salary": [9200.0, 4100.0],
})


def test_a_step_may_narrow_a_value_it_also_reads():
    """Doc 03 §3.2: 'A module may produce a value it also consumes. That is
    how a waterfall is expressed.'"""
    ceiling, income = _waterfall()
    out = flow(ceiling, income).apply(WATERFALL_FRAME)
    assert out["term_cap"].to_list() == [60.0, 48.0]


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_the_waterfall_agrees_across_modes(mode):
    ceiling, income = _waterfall()
    out = flow(ceiling, income).apply(WATERFALL_FRAME, mode=mode)
    assert out["term_cap"].to_list() == [60.0, 48.0]


def test_the_version_chain_matches_the_answer():
    """Doc 03 §7: qualification is by producing module. An audit trail that
    disagrees with the value is worse than no audit trail."""
    ceiling, income = _waterfall()
    p = flow(ceiling, income).emit("term_cap@*")
    out = p.apply(WATERFALL_FRAME)
    assert out["term_cap@product_ceiling"].to_list() == [60.0, 60.0]
    assert out["term_cap@cap_by_income"].to_list() == [60.0, 48.0]


def test_a_legitimate_name_resembling_its_output_is_not_a_build_error():
    """Doc 03 §2.2's did-you-mean must not fire on a step's own output."""
    def term_cap_a(term_cap: float) -> float:
        """Narrow."""
        return term_cap - 1.0

    out = flow(term_cap_a).apply(pl.DataFrame({"term_cap": [60.0]}))
    assert out["term_cap_a"].to_list() == [59.0]


# ---------------------------------------------------------------------------
# C — dtypes survive the boundary
# ---------------------------------------------------------------------------

def test_an_int_step_returns_an_integer_column():
    """Doc 00 §2: 'money is scaled int64'. Doc 05 §9 criterion 4."""
    def cents(rands: int) -> int:
        """To cents."""
        return rands * 100

    out = flow(cents).apply(pl.DataFrame({"rands": pl.Series([12, 34], dtype=pl.Int64)}))
    assert out["cents"].dtype == pl.Int64
    assert out["cents"].to_list() == [1200, 3400]


def test_int64_above_2_to_the_53_is_not_degraded():
    """Doc 03 §1 names Int64→float64 above 2**53 as a REJECTED design."""
    big = 2 ** 53 + 1

    def passthrough(n: int) -> int:
        """Add nothing."""
        return n + 0

    out = flow(passthrough).apply(pl.DataFrame({"n": pl.Series([big], dtype=pl.Int64)}))
    assert out["passthrough"].to_list() == [big]


def test_a_bool_step_returns_a_boolean_column():
    """Doc 05 §9 criterion 4: Boolean must round-trip."""
    def is_thin(score: float) -> bool:
        """Thin file."""
        return score < 600.0

    out = flow(is_thin).apply(pl.DataFrame({"score": [550.0, 700.0]}))
    assert out["is_thin"].dtype == pl.Boolean
    assert out["is_thin"].to_list() == [True, False]


def test_a_string_input_is_never_silently_zeroed():
    """Doc 05 §1.5: strings enter as CODES. Whatever the step sees, the answer
    must not be silently wrong."""
    def is_private(sector: str) -> float:
        """Private sector."""
        return 1.0 if sector == "private" else 0.0

    frame = pl.DataFrame({"sector": ["private", "public", "private"]})
    try:
        out = flow(is_private).apply(frame)
    except Exception:
        return  # a loud failure is acceptable; a silent wrong answer is not
    assert out["is_private"].to_list() == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# D — a misspelled param is a hard error, for every kind of model
# ---------------------------------------------------------------------------

def test_a_misspelled_param_is_rejected_for_a_hand_written_model():
    """Doc 03 §10: 'A misspelled param is a hard error (extra="forbid"), not
    silence.' §4.4: hand-written and harvested are indistinguishable."""
    from pydantic import BaseModel, Field

    class CapParams(BaseModel):
        cap: float = Field(48.0, ge=6, le=60)

    def capper(term_cap: float, params) -> float:
        """Cap."""
        return min(term_cap, params.cap)

    p = flow(module(capper, name="capper", params=CapParams))
    frame = pl.DataFrame({"term_cap": [60.0]})

    with pytest.raises(Exception):
        p.apply(frame, params={"capper": {"capp": 10.0}})   # typo


def test_a_step_reading_shared_without_shared_supplied_fails_clearly():
    """Doc 03 §4.2. The failure must name the missing field, not raise
    'Field names cannot start with an underscore'."""
    def uses_shared(x: float, shared) -> float:
        """Scale."""
        return x * shared.base_rate

    p = flow(uses_shared)
    with pytest.raises(Exception) as exc:
        p.apply(pl.DataFrame({"x": [1.0]}))
    assert "_empty" not in str(exc.value)


# ---------------------------------------------------------------------------
# E — a frame column may not shadow a computed value
# ---------------------------------------------------------------------------

def test_a_frame_column_shadowing_a_step_output_is_a_build_error():
    """Doc 03 §2.1 row 3 — 'the one error that matters'."""
    def disposable_income(net_income: float, expenses: float) -> float:
        """Disposable."""
        return net_income - expenses

    frame = pl.DataFrame({
        "net_income": [9200.0], "expenses": [3100.0],
        "disposable_income": [-999.0],          # collides with the step output
    })
    with pytest.raises(Exception) as exc:
        flow(disposable_income).apply(frame)
    assert "disposable_income" in str(exc.value)
