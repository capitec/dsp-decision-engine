"""`Struct[Item]`: a struct column as one record per row, read in a kernel."""
from __future__ import annotations

import gc
from typing import TypedDict

import numpy as np
import polars as pl
import pytest

from decider import Engine, Struct, branch, flow, frame_step, missing_as, step
from decider.exceptions import MissingInputError, WiringError
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")
COMPILED = ("stepped", "fused")
ST = pl.Struct({"income": pl.Float64, "age": pl.Int64})


class Applicant(TypedDict):
    income: float
    age: int


SCHEMA = (("income", float), ("age", int))


class WithStr(TypedDict):
    income: float
    name: str


class Mixed(TypedDict):
    income: float
    age: int
    flagged: bool


class JustIncome(TypedDict):
    income: float


class WithList(TypedDict):
    income: float
    amounts: list[float]


def afford(applicant: Struct[Applicant]) -> float:
    return applicant["income"] * 2.0 + applicant["age"]


def afford_dict(applicant: dict) -> float:
    return applicant["income"] * 2.0 + applicant["age"]


def frame(rows: list[dict | None]) -> pl.DataFrame:
    return pl.DataFrame({"applicant": rows}, schema={"applicant": ST})


FRAME = frame([{"income": 100.0, "age": 30}, {"income": 200.0, "age": 40}])


def test_two_fields_of_a_struct_agree_across_every_mode():
    out = assert_equivalent(flow(afford, name="p"), FRAME)
    assert out["afford"].to_list() == [230.0, 440.0]


@pytest.mark.parametrize("mode", COMPILED)
def test_a_struct_input_joins_the_shared_array_kernel(mode):
    exe = Engine().bind(flow(afford, name="p"), mode=mode)
    exe.run(FRAME)
    assert exe.fallbacks() == {}


def test_a_plain_dict_input_keeps_running_in_python():
    exe = Engine().bind(flow(afford_dict, name="p"), mode="fused")
    assert exe.run(FRAME)["afford_dict"].to_list() == [230.0, 440.0]
    assert "no kernel takes" in exe.fallbacks()["p/afford_dict"]


def test_float_int_and_bool_fields_all_read():
    def score_it(applicant: Struct[Mixed]) -> float:
        return applicant["income"] + applicant["age"] + (1.0 if applicant["flagged"] else 0.0)

    df = pl.DataFrame({"applicant": [{"income": 1.0, "age": 2, "flagged": True},
                                     {"income": 3.0, "age": 4, "flagged": False}]},
                      schema={"applicant": pl.Struct({"income": pl.Float64, "age": pl.Int64,
                                                      "flagged": pl.Boolean})})
    assert assert_equivalent(flow(score_it, name="p"), df)["score_it"].to_list() == [4.0, 7.0]


def test_two_steps_must_declare_the_same_fields_of_one_column():
    @step(output="doubled")
    def half(applicant: Struct[JustIncome]) -> float:
        return applicant["income"] / 2.0

    with pytest.raises(WiringError, match="an input column has one type"):
        Engine().bind(flow(half, afford, name="p"), mode="fused")


def test_two_steps_sharing_one_item_read_the_same_column():
    @step(output="doubled")
    def half(applicant: Struct[Applicant]) -> float:
        return applicant["income"] / 2.0

    out = assert_equivalent(flow(half, afford, name="p"), FRAME)
    assert out["doubled"].to_list() == [50.0, 100.0]
    assert out["afford"].to_list() == [230.0, 440.0]


def test_a_field_the_step_does_not_declare_is_ignored():
    wide = pl.DataFrame({"applicant": [{"income": 100.0, "age": 30, "extra": 9.0}]},
                        schema={"applicant": pl.Struct({"income": pl.Float64, "age": pl.Int64,
                                                        "extra": pl.Float64})})
    assert assert_equivalent(flow(afford, name="p"), wide)["afford"].to_list() == [230.0]


@pytest.mark.parametrize("mode", COMPILED)
def test_a_null_struct_names_the_column(mode):
    exe = Engine().bind(flow(afford, name="p"), mode=mode)
    with pytest.raises(MissingInputError, match="input 'applicant' of step 'p/afford'"):
        exe.run(frame([{"income": 100.0, "age": 30}, None]))
    with pytest.raises(MissingInputError, match="input 'applicant'"):
        exe.score({"applicant": None})


@pytest.mark.parametrize("mode", COMPILED)
def test_a_null_field_names_the_field(mode):
    exe = Engine().bind(flow(afford, name="p"), mode=mode)
    with pytest.raises(MissingInputError, match=r"input 'applicant\.age'"):
        exe.run(frame([{"income": 100.0, "age": None}]))
    with pytest.raises(MissingInputError, match=r"input 'applicant\.age'"):
        exe.score({"applicant": {"income": 100.0, "age": None}})


@pytest.mark.parametrize("mode", MODES)
def test_an_absent_struct_column_says_it_is_absent(mode):
    exe = Engine().bind(flow(afford, name="p"), mode=mode)
    with pytest.raises(MissingInputError, match="is not in the input frame or record"):
        exe.score({})
    with pytest.raises(MissingInputError, match="is not in the input frame or record"):
        exe.run(pl.DataFrame({"other": [1.0]}))


def test_a_record_without_a_declared_field_names_the_field():
    exe = Engine().bind(flow(afford, name="p"), mode="fused")
    with pytest.raises(ValueError, match="has no field 'age' on row 0"):
        exe.score({"applicant": {"income": 100.0}})


def test_a_column_without_a_declared_field_names_the_field():
    exe = Engine().bind(flow(afford, name="p"), mode="fused")
    narrow = pl.DataFrame({"applicant": [{"income": 100.0}]},
                          schema={"applicant": pl.Struct({"income": pl.Float64})})
    with pytest.raises(ValueError, match=r"declares field\(s\) \['age'\]"):
        exe.run(narrow)


def test_a_column_that_is_not_a_struct_says_so():
    exe = Engine().bind(flow(afford, name="p"), mode="fused")
    with pytest.raises(TypeError, match="holds float64 values, not structs"):
        exe.run(pl.DataFrame({"applicant": [1.0]}))


def test_a_field_type_no_record_holds_keeps_the_step_in_python():
    def afford_str(applicant: Struct[WithStr]) -> float:
        return applicant["income"] * 2.0

    df = pl.DataFrame({"applicant": [{"income": 100.0, "name": "a"}]},
                      schema={"applicant": pl.Struct({"income": pl.Float64, "name": pl.String})})
    exe = Engine().bind(flow(afford_str, name="p"), mode="fused")
    with pytest.warns(UserWarning, match=r"reads field 'applicant\.name' as <class 'str'>"):
        assert exe.run(df)["afford_str"].to_list() == [200.0]


def test_a_nested_field_is_refused_at_bind_and_points_at_dict():
    def afford_list(applicant: Struct[WithList]) -> float:
        return applicant["income"] + sum(applicant["amounts"])

    with pytest.raises(TypeError, match=r"field 'amounts' is list\[float\].*as dict"):
        Engine().bind(flow(afford_list, name="p"), mode="fused")


def test_a_struct_reaches_a_step_that_runs_one_compiled_call_per_row():
    def label(applicant: Struct[Applicant]) -> str:
        return "hi" if applicant["income"] > 150.0 else "lo"

    out = assert_equivalent(flow(label, name="p"), FRAME)
    assert out["label"].to_list() == ["lo", "hi"]


@pytest.mark.parametrize("mode", MODES)
def test_dividing_by_zero_still_raises_on_the_per_row_path(mode):
    # A record's fields are numpy scalars, so only a compiled dispatcher is ever handed one:
    # numba's python error model raises here exactly as a Python body does.
    def label(applicant: Struct[Applicant]) -> str:
        return "hi" if applicant["income"] / applicant["age"] > 1.0 else "lo"

    exe = Engine().bind(flow(label, name="p"), mode=mode)
    with pytest.raises(ZeroDivisionError):
        exe.run(frame([{"income": 1.0, "age": 0}]))


def test_a_fill_keeps_the_step_in_python():
    def afford_fill(applicant: Struct[Applicant] = missing_as({"income": 0.0, "age": 0})) -> float:
        return applicant["income"] * 2.0 + applicant["age"]

    exe = Engine().bind(flow(afford_fill, name="p"), mode="fused")
    with pytest.warns(UserWarning, match="which a kernel record can't hold"):
        assert exe.run(frame([{"income": 100.0, "age": 30}, None]))["afford_fill"].to_list() == [230.0, 0.0]


def test_a_branch_reads_the_rows_of_its_own_arm():
    def rich(applicant: Struct[Applicant]) -> bool:
        return applicant["income"] > 150.0

    @step(output="amount")
    def double(applicant: Struct[Applicant]) -> float:
        return applicant["income"] * 2.0

    @step(output="amount")
    def triple(applicant: Struct[Applicant]) -> float:
        return applicant["income"] * 3.0

    pipeline = flow(branch(rich, double, triple, modifies=["amount"], name="by"), name="p")
    out = assert_equivalent(pipeline, FRAME)
    assert out["amount"].to_list() == [300.0, 400.0]


def test_a_null_struct_on_a_row_another_arm_handles_is_not_an_error():
    def wanted(flag: bool) -> bool:
        return flag

    @step(output="amount")
    def uses(applicant: Struct[Applicant]) -> float:
        return applicant["income"] * 2.0 + applicant["age"]

    @step(output="amount")
    def zero(flag: bool) -> float:
        return 0.0

    df = pl.DataFrame({"applicant": [{"income": 100.0, "age": 30}, None], "flag": [True, False]},
                      schema={"applicant": ST, "flag": pl.Boolean})
    pipeline = flow(branch(wanted, uses, zero, modifies=["amount"], name="by"), name="p")
    assert assert_equivalent(pipeline, df)["amount"].to_list() == [230.0, 0.0]


def _big(n: int) -> pl.DataFrame:
    # Above FrameView's borrow threshold, so the boundary hands back views of Arrow memory.
    income = np.arange(n, dtype=np.float64)
    return pl.DataFrame({"income": income, "age": np.arange(n, dtype=np.int64) % 70}).select(
        pl.struct(["income", "age"]).alias("applicant"))


def test_records_stay_valid_after_the_arrow_export_is_released():
    n = 5000
    df = _big(n)
    expected = (df["applicant"].struct.field("income").to_numpy() * 2.0
                + df["applicant"].struct.field("age").to_numpy())
    exe = Engine().bind(flow(afford, name="p"), mode="fused")
    state, params = exe.prepare(df)
    del df
    for _ in exe.runner.iterate(exe.plan, state, params):
        # Frees every Arrow export the boundary no longer holds; a record that
        # borrowed one instead of copying reads freed memory from here on.
        gc.collect()
    assert state.column("afford").to_numpy().tolist() == expected.tolist()
    records = state.representation(state.versions("applicant")[0], ("struct", SCHEMA), None)
    assert records.base is None, "the records must own their memory, not borrow Arrow's"


def test_a_struct_reaches_the_kernel_through_the_frame_path_too():
    @frame_step(reads=["applicant"], writes=["seen"])
    def mark(df):
        return df.with_columns(seen=pl.lit(1.0))

    exe = Engine().bind(flow(mark, afford, name="p"), mode="fused")
    assert exe.score({"applicant": {"income": 100.0, "age": 30}})["afford"] == 230.0


def test_an_overridden_struct_column_is_read_from_the_new_values():
    exe = Engine().bind(flow(afford, name="p"), mode="fused")
    session = exe.session(FRAME)
    session.set("applicant", [{"income": 1.0, "age": 2}, {"income": 3.0, "age": 4}])
    session.resume()
    assert session.state.column("afford").to_list() == [4.0, 10.0]
