"""Every value shape a step can read or write, in all three modes: the answer, and where it runs."""
from __future__ import annotations

import warnings
from datetime import date
from typing import TypedDict

import polars as pl
import pytest

from decider import Engine, Raw, Rows, Struct, flow, missing_as, raw_str
from decider.exceptions import FallbackWarning
from decider.testing import assert_equivalent

COMPILED = ("stepped", "fused")
PRIVATE = raw_str("private")


class Account(TypedDict):
    balance: float


class Applicant(TypedDict):
    income: float
    dependants: int


class DatedAccount(TypedDict):
    balance: float
    opened: date


class CountedAccount(TypedDict):
    balance: float
    count: int | None


class NamedApplicant(TypedDict):
    income: float
    name: str


FRAME = pl.DataFrame(
    {"sector": ["private", "public"],
     "blob": [b"private", b"public"],
     "accounts": [[{"balance": 10.0}, {"balance": 5.0}], [{"balance": 3.0}]],
     "applicant": [{"income": 100.0, "dependants": 2}, {"income": 50.0, "dependants": 0}],
     "history": [[1.0, 2.0], [3.0]]},
    schema_overrides={"applicant": pl.Struct({"income": pl.Float64, "dependants": pl.Int64})},
)
RECORD = {"sector": "private", "blob": b"private", "accounts": [{"balance": 10.0}, {"balance": 5.0}],
          "applicant": {"income": 100.0, "dependants": 2}, "history": [1.0, 2.0]}


def semantic_str(sector: str) -> float:
    return 1.0 if sector == "private" else 0.0


def raw_str_code(sector: Raw[str]) -> float:
    return 1.0 if sector == PRIVATE else 0.0


def semantic_bytes(blob: bytes) -> float:
    return 1.0 if blob == b"private" else 0.0


def raw_bytes_span(sector: Raw[bytes]) -> float:
    return 1.0 if sector == "private" else 0.0


def raw_bytes_prefix(sector: Raw[bytes]) -> float:
    return float(len(sector)) if sector.startswith("priv") else 0.0


def list_of_dicts(accounts: list[dict]) -> float:
    return sum(a["balance"] for a in accounts)


def rows_of_items(accounts: Rows[Account]) -> float:
    total = 0.0
    for j in range(len(accounts.balance)):
        total += accounts.balance[j]
    return total


def plain_dict(applicant: dict) -> float:
    return applicant["income"] - applicant["dependants"] * 10.0


def struct_record(applicant: Struct[Applicant]) -> float:
    return applicant["income"] - applicant["dependants"] * 10.0


def list_of_floats(history: list[float]) -> float:
    return sum(history)


# (step, its answer over FRAME, its answer for RECORD, whether a kernel holds it)
SHAPES = [
    (semantic_str, [1.0, 0.0], 1.0, False),
    (raw_str_code, [1.0, 0.0], 1.0, True),
    (semantic_bytes, [1.0, 0.0], 1.0, False),
    (raw_bytes_span, [1.0, 0.0], 1.0, True),
    (raw_bytes_prefix, [7.0, 0.0], 7.0, True),
    (list_of_dicts, [15.0, 3.0], 15.0, False),
    (rows_of_items, [15.0, 3.0], 15.0, True),
    (plain_dict, [80.0, 50.0], 80.0, False),
    (struct_record, [80.0, 50.0], 80.0, True),
    (list_of_floats, [3.0, 3.0], 3.0, False),
]
IDS = [fn.__name__ for fn, *_ in SHAPES]


@pytest.fixture(autouse=True)
def _quiet():
    # Where a shape runs is asserted below; the warning itself is covered in test_compiled_fallback.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FallbackWarning)
        yield


@pytest.mark.parametrize("fn, over_frame, for_record, _kernel", SHAPES, ids=IDS)
def test_every_shape_gives_the_same_answer_in_every_mode(fn, over_frame, for_record, _kernel):
    assert assert_equivalent(flow(fn, name="p"), FRAME)[fn.__name__].to_list() == over_frame


@pytest.mark.parametrize("fn, _over_frame, for_record, _kernel", SHAPES, ids=IDS)
@pytest.mark.parametrize("mode", ("interpreted", *COMPILED))
def test_every_shape_scores_one_record(mode, fn, _over_frame, for_record, _kernel):
    assert Engine().bind(flow(fn, name="p"), mode=mode).score(RECORD)[fn.__name__] == for_record


@pytest.mark.parametrize("fn, _over_frame, _for_record, kernel", SHAPES, ids=IDS)
@pytest.mark.parametrize("mode", COMPILED)
def test_only_the_shapes_a_kernel_holds_stay_in_it(mode, fn, _over_frame, _for_record, kernel):
    exe = Engine().bind(flow(fn, name="p"), mode=mode)
    exe.run(FRAME)
    assert (exe.fallbacks() == {}) is kernel, exe.fallbacks()


def every_shape_at_once(sector: str, accounts: list[dict], applicant: dict, history: list[float]) -> float:
    return (1.0 if sector == "private" else 0.0) + sum(a["balance"] for a in accounts) \
        + applicant["income"] + sum(history)


def every_compiled_shape_at_once(sector: Raw[str], accounts: Rows[Account],
                                 applicant: Struct[Applicant]) -> float:
    total = 0.0
    for j in range(len(accounts.balance)):
        total += accounts.balance[j]
    return (1.0 if sector == PRIVATE else 0.0) + total + applicant["income"]


def test_the_python_shapes_combine_in_one_step():
    out = assert_equivalent(flow(every_shape_at_once, name="p"), FRAME)
    assert out["every_shape_at_once"].to_list() == [119.0, 56.0]


def test_the_opt_in_shapes_combine_in_one_step():
    out = assert_equivalent(flow(every_compiled_shape_at_once, name="p"), FRAME)
    assert out["every_compiled_shape_at_once"].to_list() == [116.0, 53.0]


def test_a_raw_bytes_input_over_a_bytes_column_is_refused_the_same_way_in_every_mode():
    # `Raw[bytes]` is a string column read as its UTF-8 bytes; a real bytes column is `bytes`.
    def over_blob(blob: Raw[bytes]) -> float:
        return float(len(blob))

    for mode in ("interpreted", *COMPILED):
        with pytest.raises(TypeError, match="'blob' is a string input"):
            Engine().bind(flow(over_blob, name="p"), mode=mode).run(FRAME)


def test_a_step_mixing_an_opt_in_and_a_python_shape_still_agrees():
    def mixed(sector: Raw[str], accounts: list[dict]) -> float:
        return (1.0 if sector == PRIVATE else 0.0) + sum(a["balance"] for a in accounts)

    out = assert_equivalent(flow(mixed, name="p"), FRAME)
    assert out["mixed"].to_list() == [16.0, 3.0]


def test_each_shape_chains_into_the_next_step():
    def widen(sector: str) -> bytes:
        return sector.encode()

    def measure(widen: bytes) -> float:
        return float(len(widen))

    out = assert_equivalent(flow(widen, measure, name="p"), FRAME)
    assert out["measure"].to_list() == [7.0, 6.0]


# --- nulls: `| None` hands the step a real None, `missing_as` fills before it runs ---------------

NULLABLE = pl.DataFrame(
    {"sector": ["private", None],
     "accounts": [[{"balance": 1.0}], None],
     "applicant": [{"income": 5.0}, None]},
    schema_overrides={"applicant": pl.Struct({"income": pl.Float64})},
)


def optional_str(sector: str | None) -> float:
    return 0.0 if sector is None else 1.0


def optional_code(sector: Raw[str] | None) -> float:
    return 0.0 if sector is None else 1.0


def optional_span(sector: Raw[bytes] | None) -> float:
    return float(len(sector))


def optional_list(accounts: list[dict] | None) -> float:
    return 0.0 if accounts is None else float(len(accounts))


def optional_rows(accounts: Rows[Account] | None) -> float:
    return 0.0 if accounts is None else float(len(accounts.balance))


def optional_dict(applicant: dict | None) -> float:
    return 0.0 if applicant is None else applicant["income"]


def optional_struct(applicant: Struct[Applicant] | None) -> float:
    return 0.0 if applicant is None else applicant["income"]


def filled_str(sector: str = missing_as("none")) -> float:
    return 1.0 if sector == "none" else 0.0


def filled_list(accounts: list[dict] = missing_as([])) -> float:
    return float(len(accounts))


def filled_rows(accounts: Rows[Account] = missing_as([])) -> float:
    return float(len(accounts.balance))


NULLS = [(optional_str, [1.0, 0.0]), (optional_code, [1.0, 0.0]), (optional_span, [7.0, 0.0]),
         (optional_list, [1.0, 0.0]), (optional_rows, [1.0, 0.0]), (optional_dict, [5.0, 0.0]),
         (optional_struct, [5.0, 0.0]), (filled_str, [0.0, 1.0]), (filled_list, [1.0, 0.0]),
         (filled_rows, [1.0, 0.0])]


@pytest.mark.parametrize("fn, expected", NULLS, ids=[fn.__name__ for fn, _ in NULLS])
def test_a_null_reads_the_same_way_in_every_mode(fn, expected):
    assert assert_equivalent(flow(fn, name="p"), NULLABLE)[fn.__name__].to_list() == expected


@pytest.mark.parametrize("fn, expected", NULLS, ids=[fn.__name__ for fn, _ in NULLS])
@pytest.mark.parametrize("mode", ("interpreted", *COMPILED))
def test_a_null_reads_the_same_way_for_one_record(mode, fn, expected):
    record = {"sector": None, "accounts": None, "applicant": None}
    assert Engine().bind(flow(fn, name="p"), mode=mode).score(record)[fn.__name__] == expected[1]


# --- outputs ------------------------------------------------------------------------------------

def out_str(sector: str) -> str:
    return sector.upper()


def out_code(sector: Raw[str]) -> Raw[str]:
    return sector


def out_list(sector: str) -> list[float]:
    return [float(len(sector))]


def out_dict(sector: str) -> dict:
    return {"n": len(sector)}


OUTPUTS = [(out_str, ["PRIVATE", "PUBLIC"]), (out_code, [PRIVATE, raw_str("public")]),
           (out_list, [[7.0], [6.0]]), (out_dict, [{"n": 7}, {"n": 6}])]


@pytest.mark.parametrize("fn, expected", OUTPUTS, ids=[fn.__name__ for fn, _ in OUTPUTS])
def test_every_output_shape_comes_back_the_same_in_every_mode(fn, expected):
    assert assert_equivalent(flow(fn, name="p"), FRAME)[fn.__name__].to_list() == expected


# --- what each opt-in refuses, naming the field ------------------------------------------------

def test_a_field_no_kernel_can_hold_is_refused_and_says_what_to_do():
    def total(accounts: Rows[DatedAccount]) -> float:
        return float(len(accounts.balance))

    with pytest.raises(TypeError, match="'opened'"):
        Engine().bind(flow(total, name="p"), mode="fused").run(FRAME)


def test_an_int_or_none_field_in_an_item_is_refused_because_it_has_no_null_value():
    def total(accounts: Rows[CountedAccount]) -> float:
        return float(len(accounts.balance))

    with pytest.raises(TypeError, match=r"'count'.*float \| None"):
        Engine().bind(flow(total, name="p"), mode="fused").run(FRAME)


def test_a_string_field_in_a_struct_keeps_the_step_in_python_naming_the_field():
    def income(applicant: Struct[NamedApplicant]) -> float:
        return applicant["income"]

    frame = pl.DataFrame({"applicant": [{"income": 5.0, "name": "a"}]},
                         schema={"applicant": pl.Struct({"income": pl.Float64, "name": pl.String})})
    exe = Engine().bind(flow(income, name="p"), mode="fused")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FallbackWarning)
        assert exe.run(frame)["income"].to_list() == [5.0]
    assert "applicant.name" in exe.fallbacks()["p/income"]


class MaybeAccount(TypedDict):
    balance: float | None


def test_an_optional_item_field_reads_a_null_as_nan_in_every_mode():
    def first(accounts: Rows[MaybeAccount]) -> float:
        return accounts.balance[0]

    # polars takes a struct field's type from the first row, so a leading null would type the
    # whole column Null; the value goes first.
    frame = pl.DataFrame({"accounts": [[{"balance": 2.0}], [{"balance": None}]]})
    out = assert_equivalent(flow(first, name="p"), frame)["first"].to_list()
    assert out[0] == 2.0 and out[1] != out[1]      # the value, then NaN


def test_a_null_item_field_that_is_not_optional_names_the_item_and_the_row():
    def total(accounts: Rows[Account]) -> float:
        return accounts.balance[0]

    frame = pl.DataFrame({"accounts": [[{"balance": 1.0}], [{"balance": None}]]})
    with pytest.raises(ValueError, match="'balance' is null in item 0 of row 1"):
        Engine().bind(flow(total, name="p"), mode="fused").run(frame)


def out_bytes(sector: str) -> bytes:
    return sector.encode()


def out_list_of_dicts(sector: str) -> list[dict]:
    return [{"n": len(sector)}]


@pytest.mark.parametrize("fn, expected", [(out_bytes, [b"private", b"public"]),
                                          (out_list_of_dicts, [[{"n": 7}], [{"n": 6}]])],
                         ids=["out_bytes", "out_list_of_dicts"])
def test_the_remaining_output_shapes_come_back_the_same_in_every_mode(fn, expected):
    assert assert_equivalent(flow(fn, name="p"), FRAME)[fn.__name__].to_list() == expected
