"""The testing helpers: every mode, a record run and a session agree; boundary corpus; no recompiles."""
import decimal as stdlib

import polars as pl
import pytest
from numba.extending import overload

from decikit import flow, missing_as, param
from decikit.ops import decimal
from decikit.ops.decimal import Currency
from decikit._engine import Engine
from decikit._engine.debug import Session
from decikit._engine.run import Executable
from decikit.lab.testing import assert_equivalent, corpus, no_recompile


# Nothing numba and CPython could disagree on (no division, no rounding), so
# the boundary corpus is a genuine positive case for every null policy and dtype.
def net(a: float, b: float) -> float:
    return a - b


def scaled(net: float, factor: float = param(2.0, ge=0)) -> float:
    return net * factor


def big(n: int | None) -> bool:
    return False if n is None else n > 1000


def padded(extra: float = missing_as(10.0)) -> float:
    return extra + 1.0


def flagged(active: bool) -> float:
    return 1.0 if active else 0.0


pipeline = flow(net, scaled, big, padded, flagged)

FRAME = pl.DataFrame({
    "a": [10.0, 20.0, 5.0],
    "b": [3.0, 1.0, 5.0],
    "n": [500, 2000, None],
    "extra": [1.0, None, 3.0],
    "active": [True, False, True],
})


# --- assert_equivalent ---------------------------------------------------------


def test_assert_equivalent_passes_and_returns_the_output_when_every_mode_agrees():
    out = assert_equivalent(pipeline, FRAME, params={"scaled": {"factor": 3.0}})
    assert out["scaled"].to_list() == [21.0, 57.0, 0.0]
    assert out["padded"].to_list() == [2.0, 11.0, 4.0]


@pytest.mark.parametrize("shape", ["boundary", "single", "chunked", "empty"])
def test_assert_equivalent_passes_over_every_corpus_frame(shape):
    assert_equivalent(pipeline, corpus(pipeline)[shape])


def test_assert_equivalent_needs_a_dataframe():
    with pytest.raises(TypeError, match="polars DataFrame"):
        assert_equivalent(pipeline, FRAME.to_dict())


def always_broken(x: float) -> float:
    if x > -1e300:
        raise RuntimeError("deliberately broken in every mode")
    return x


def test_assert_equivalent_fails_when_every_mode_raises_the_same_error():
    # Identical crashes in every mode are not agreement: nothing was compared.
    with pytest.raises(RuntimeError, match="deliberately broken"):
        assert_equivalent(flow(always_broken), pl.DataFrame({"x": [1.0, 2.0]}))


def bump(x):
    return x + 1.0


# Compiled code calls this instead of `bump`: a step whose semantics numba changes, on purpose.
@overload(bump)
def _bump_compiled(x):
    return lambda x: x + 2.0


def bumped(x: float) -> float:
    return bump(x)


def test_a_mode_that_computes_differently_fails_naming_the_mode_and_both_values():
    with pytest.raises(AssertionError, match="stepped mode differs from interpreted") as e:
        assert_equivalent(flow(bumped), pl.DataFrame({"x": [1.0, 5.0]}))
    assert "2.0" in str(e.value) and "3.0" in str(e.value)


def ratio(x: float, y: float) -> float:
    return x / y


@pytest.mark.parametrize("mode", ("interpreted", "stepped", "fused"))
def test_dividing_by_zero_agrees_across_modes(mode):
    with pytest.raises(ZeroDivisionError):
        Engine().bind(flow(ratio), mode).run(pl.DataFrame({"x": [1.0, 0.0], "y": [0.0, 0.0]}))


def test_a_record_run_disagreeing_with_a_frame_run_fails_naming_the_row(monkeypatch):
    run_record = Executable._run_record

    def off_by_one(self, record, params=None, trace=None):
        out = run_record(self, record, params, trace)
        return out | {"scaled": out["scaled"] + (record["a"] == 20.0)}

    monkeypatch.setattr(Executable, "_run_record", off_by_one)
    with pytest.raises(AssertionError, match=r"run\(record\) of row 1 in interpreted mode differs.*'scaled'.*39.0.*38.0"):
        assert_equivalent(pipeline, FRAME)


def test_a_session_disagreeing_with_run_fails(monkeypatch):
    output = Session.output

    def stale(self):
        out = output(self)
        return out.with_columns(pl.col("scaled") * 2) if type(self.executable.runner).__name__ == "FusedRunner" else out

    monkeypatch.setattr(Session, "output", stale)
    with pytest.raises(AssertionError, match="session resumed to the end in fused mode differs from run"):
        assert_equivalent(pipeline, FRAME)


def _override_before_net(s):
    s.break_at("net")
    s.resume()
    s.set("a", 100.0)


def test_a_scripted_break_set_resume_agrees_across_modes():
    out = assert_equivalent(pipeline, FRAME, script=_override_before_net)
    assert out["scaled"].to_list() == [14.0, 38.0, 0.0]  # the plain run; the override only lives in the sessions


def test_a_scripted_session_diverging_across_modes_fails():
    def per_mode(s):
        _override_before_net(s)
        if type(s.executable.runner).__name__ == "SteppedRunner":
            s.set("a", 0.0)

    with pytest.raises(AssertionError, match="scripted session in stepped mode differs from interpreted"):
        assert_equivalent(pipeline, FRAME, script=per_mode)


@pytest.mark.xfail(strict=True, reason="by design: a fused session pauses before the whole kernel, so a set lands before its earlier steps run")
def test_a_breakpoint_on_a_later_step_of_a_fused_kernel_agrees_across_modes():
    def override_scaled_input(s):
        s.break_at("scaled")
        s.resume()
        s.set("net", 1.0)

    assert_equivalent(pipeline, FRAME, script=override_scaled_input)


def test_modes_can_be_narrowed():
    out = assert_equivalent(flow(bumped), pl.DataFrame({"x": [1.0]}), modes=("stepped", "fused"))
    assert out["bumped"].to_list() == [3.0]


# --- corpus ----------------------------------------------------------------------


def test_corpus_frames_share_one_schema_and_the_expected_row_counts():
    frames = corpus(pipeline)
    assert set(frames) == {"boundary", "single", "chunked", "empty"}
    assert all(f.schema == frames["boundary"].schema for f in frames.values())
    assert (frames["single"].height, frames["empty"].height) == (1, 0)
    assert frames["chunked"].n_chunks() == 2
    assert frames["chunked"].equals(frames["boundary"])


def test_corpus_has_a_column_per_input_typed_by_its_declaration():
    schema = corpus(pipeline)["boundary"].schema
    assert dict(schema) == {"a": pl.Float64, "b": pl.Float64, "n": pl.Int64, "extra": pl.Float64,
                            "active": pl.Boolean, "case": pl.String}


def _case(name):
    boundary = corpus(pipeline)["boundary"]
    return boundary.filter(pl.col("case") == name)


def test_corpus_zero_and_negative_rows_move_only_the_named_column():
    zero_a, negative_a, baseline = _case("zero:a"), _case("negative:a"), _case("baseline")
    assert zero_a["a"].to_list() == [0.0] and negative_a["a"].item() < 0
    assert zero_a.drop("a", "case").equals(baseline.drop("a", "case"))


def test_corpus_skips_negative_for_a_bool_but_not_for_an_int():
    cases = corpus(pipeline)["boundary"]["case"].to_list()
    assert "negative:active" not in cases and "zero:active" in cases
    assert "negative:n" in cases
    assert _case("zero:active")["active"].item() is False


def test_corpus_gives_an_int_input_a_value_past_exact_float64():
    assert _case("int_near_2**53:n")["n"].item() == 2**53 + 1
    assert _case("int_near_2**53:a").height == 0


def test_corpus_nulls_only_inputs_no_step_requires():
    cases = corpus(pipeline)["boundary"]["case"].to_list()
    assert {c for c in cases if c.startswith("null:")} == {"null:n", "null:extra"}
    assert _case("null:extra")["extra"].item() is None


# --- no_recompile ----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_no_recompile_passes_for_a_value_only_retune(mode):
    exe = Engine().bind(pipeline, mode=mode)
    exe.run(FRAME)
    exe.run(FRAME.row(0, named=True))
    with no_recompile():
        exe.run(FRAME, params={"scaled": {"factor": 3.0}})
        exe.run(FRAME.row(0, named=True), params={"scaled": {"factor": 4.0}})


def test_no_recompile_fails_when_something_compiles():
    offset = 7.25  # a closure: never served from numba's disk cache, so it always compiles here

    def shifted(x: float) -> float:
        return x + offset

    with pytest.raises(AssertionError, match="numba compiled .*shifted"):
        with no_recompile():
            Engine().bind(flow(shifted), mode="stepped").run(pl.DataFrame({"x": [1.0]}))


def test_corpus_adds_rounding_ties_for_every_float_input():
    cases = corpus(pipeline)["boundary"]["case"].to_list()
    assert "tie:2.675:a" in cases and "tie:-2.5:b" in cases
    assert not any(c.startswith("tie:") and c.endswith(":n") for c in cases)
    assert _case("tie:2.675:a")["a"].item() == 2.675


def interest(principal: Currency, count: int, rate: float = param(0.0875)) -> Currency:
    return decimal.round(principal * rate, 2) * count


def test_corpus_gives_a_decimal_input_a_decimal_column_with_large_amounts():
    boundary = corpus(flow(interest))["boundary"]
    cases = boundary["case"].to_list()
    assert boundary.schema["principal"] == pl.Decimal(38, 4)
    assert {c for c in cases if c.startswith("large:")} == {
        f"large:{k}:principal" for k in ("10**11", "10**13", "2**62//1000")}
    assert boundary.filter(pl.col("case") == "large:10**13:principal")["principal"].item() == stdlib.Decimal(10**13).scaleb(-4)
    assert boundary.filter(pl.col("case") == "int_near_2**53:principal")["principal"].item() == stdlib.Decimal(2**53 + 1).scaleb(-4)
    assert {"zero:principal", "negative:principal", "int_near_2**53:count"} <= set(cases)


def test_every_mode_agrees_over_the_money_corpus():
    flow_ = flow(interest)
    for frame in corpus(flow_).values():
        assert_equivalent(flow_, frame)


def test_corpus_of_a_step_with_no_inputs_runs_in_every_mode():
    def constant() -> float:
        return 1.0

    frames = corpus(flow(constant))
    assert frames["boundary"].columns == ["case"] and frames["boundary"].height == 1
    for frame in frames.values():
        assert_equivalent(flow(constant), frame)
