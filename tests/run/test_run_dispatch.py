"""`run` takes a dict record, a polars DataFrame or a LazyFrame, and nothing else."""
from types import MappingProxyType

import polars as pl
import pytest

from decikit import flow


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordable(disposable_income: float) -> bool:
    return disposable_income > 2000.0


pipeline = flow(disposable_income, affordable)
FRAME = pl.DataFrame({"net_income": [4100.0, 2500.0], "expenses": [1500.0, 1000.0]})


def test_a_dict_in_gives_a_dict_out(bind):
    assert bind(pipeline).run({"net_income": 4100.0, "expenses": 1500.0}) == {
        "net_income": 4100.0, "expenses": 1500.0, "affordable": True}


def test_any_mapping_takes_the_record_path(bind):
    assert bind(pipeline).run(MappingProxyType({"net_income": 4100.0, "expenses": 1500.0}))["affordable"] is True


def test_a_frame_in_gives_a_frame_out(bind):
    out = bind(pipeline).run(FRAME)
    assert isinstance(out, pl.DataFrame)
    assert out["affordable"].to_list() == [True, False]


def test_a_lazy_frame_in_gives_the_collected_frame_out(bind):
    exe = bind(pipeline)
    out = exe.run(FRAME.lazy())
    assert isinstance(out, pl.DataFrame)
    assert out.equals(exe.run(FRAME))


def test_a_list_is_refused_naming_the_accepted_types(bind):
    with pytest.raises(TypeError, match="takes a dict record, a polars DataFrame or LazyFrame, not list"):
        bind(pipeline).run([{"net_income": 4100.0, "expenses": 1500.0}])


def test_a_dict_record_equals_the_matching_row_of_a_frame_run(bind):
    exe = bind(pipeline)
    rows = exe.run(FRAME).iter_rows(named=True)
    assert [exe.run(record) for record in FRAME.iter_rows(named=True)] == list(rows)
