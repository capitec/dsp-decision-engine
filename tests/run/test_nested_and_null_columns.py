import datetime as dt
import warnings

import polars as pl
import pytest

from decider import Engine, flow, frame_step, missing_as, param


def _both(pipeline, record):
    exe = Engine().bind(pipeline)
    return exe.run(pl.DataFrame([record])).row(0, named=True), exe.score(record)


def test_list_of_struct_frame_output_materialises():
    @frame_step(reads=["x"], writes=["z"])
    def fs(df):
        return df.with_columns(pl.Series("z", [[{"a": 1}]] * len(df)))

    run, score = _both(flow(fs, name="p"), {"x": 1.0})
    assert run["z"] == score["z"] == [{"a": 1}]


def test_list_of_struct_passes_from_one_frame_step_to_another():
    @frame_step(reads=["x"], writes=["accts"])
    def merge(df):
        return df.with_columns(pl.Series("accts", [[{"amt": 1.0}, {"amt": 2.0}]] * len(df)))

    @frame_step(reads=["accts"], writes=["total"])
    def total(df):
        return df.with_columns(total=pl.col("accts").list.eval(pl.element().struct.field("amt")).list.sum())

    out = Engine().bind(flow(merge, total, name="p")).run(pl.DataFrame({"x": [1.0, 2.0]}))
    assert out["total"].to_list() == [3.0, 3.0]


def test_list_of_list_frame_output_stays_nested_lists():
    @frame_step(reads=["x"], writes=["z"])
    def fl(df):
        return df.with_columns(pl.Series("z", [[[1, 2], [3]]] * len(df)))

    out = Engine().bind(flow(fl, name="p")).run(pl.DataFrame({"x": [1.0]}))
    assert out.schema["z"] == pl.List(pl.List(pl.Int64))
    assert out["z"].to_list() == [[[1, 2], [3]]]


def test_list_of_date_stays_dates_between_frame_steps():
    days = [dt.date(2026, 1, 1), dt.date(2025, 6, 1)]
    seen = {}

    @frame_step(reads=["x"], writes=["ev_date"])
    def s1(df):
        return df.with_columns(pl.Series("ev_date", [days] * len(df)))

    @frame_step(reads=["ev_date"], writes=["seen"])
    def s2(df):
        seen["dtype"], seen["value"] = df.schema["ev_date"], df["ev_date"].to_list()[0]
        return df.with_columns(seen=pl.lit(1))

    Engine().bind(flow(s1, s2, name="p")).run(pl.DataFrame({"x": [1.0]}))
    assert seen == {"dtype": pl.List(pl.Date), "value": days}


def test_list_of_str_with_empty_rows_materialises():
    @frame_step(reads=["x"], writes=["y"])
    def f(df):
        return df.with_columns(pl.Series("y", [["a", "b"], [], ["c"]], dtype=pl.List(pl.Utf8)))

    out = Engine().bind(f).run(pl.DataFrame({"x": [1, 2, 3]}))
    assert out["y"].to_list() == [["a", "b"], [], ["c"]]


def test_list_of_struct_input_reaches_a_plain_step_as_dicts():
    def first(accounts: list[dict]) -> str:
        return repr(accounts[0])

    @frame_step(reads=["first"], writes=["n"])
    def f(df):
        return df.with_columns(n=pl.col("first").str.len_chars())

    account = {"opened": dt.date(2020, 1, 1), "q": 1.0}
    run, score = _both(flow(first, f, name="p").emit("first"), {"accounts": [account]})
    assert run["first"] == score["first"] == repr(account)


def test_list_input_is_a_list_under_run_and_score():
    def kind(hist: list[float]) -> str:
        return type(hist).__name__

    run, score = _both(flow(kind, name="p"), {"hist": [1.0, 2.0]})
    assert run["kind"] == score["kind"] == "list"


@pytest.mark.parametrize("record, n", [
    ({"hist": [1.0], "last": None, "v": 1.0}, 2),
    ({"hist": [], "last": dt.date(2026, 1, 1), "v": None}, 1),
])
def test_one_record_with_empty_list_or_lone_none_scores_with_a_frame_step(record, n):
    def count(hist: list[float], last: dt.date | None = None, v: float | None = None) -> int:
        return len(hist) + (last is not None) + (v is not None)

    @frame_step(reads=["count"], writes=["m"])
    def fm(df):
        return df.with_columns(m=pl.col("count") + 1)

    out = Engine().bind(flow(count, fm, name="p")).score(record)
    assert out["m"] == n + 1


@pytest.mark.parametrize("with_frame_step", [False, True])
def test_missing_as_empty_list_fills_each_absent_row(with_frame_step):
    def total(hist: list[float] = missing_as([])) -> float:
        hist.append(1.0)  # must not leak into the declared fill or other rows
        return float(len(hist))

    @frame_step(reads=["total"], writes=["t2"])
    def t2(df):
        return df.with_columns(t2=pl.col("total") * 2)

    exe = Engine().bind(flow(total, t2, name="p").emit("total") if with_frame_step else flow(total, name="p"))
    assert exe.score({"x": 1.0})["total"] == 1.0
    out = exe.run(pl.DataFrame({"hist": [[5.0], None, None]}))
    assert out["total"].to_list() == [2.0, 1.0, 1.0]


def test_list_of_float_output_keeps_floats_whatever_the_first_row():
    def rates(x: float) -> list[float]:
        return [1, 2] if x < 2 else [1.5]

    out = Engine().bind(flow(rates, name="p")).run(pl.DataFrame({"x": [1.0, 2.0]}))
    assert out.schema["rates"] == pl.List(pl.Float64)
    assert out["rates"].to_list() == [[1.0, 2.0], [1.5]]


def test_param_name_in_the_record_warns_once():
    def intensity(base: int, stack_enabled: bool = param(True)) -> int:
        return base + 1 if stack_enabled else base

    exe = Engine().bind(flow(intensity, name="p"))
    with pytest.warns(UserWarning, match="'stack_enabled' is a param.*params document"):
        exe.score({"base": 2, "stack_enabled": False})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exe.score({"base": 2, "stack_enabled": False})
