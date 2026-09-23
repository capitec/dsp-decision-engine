"""The boundary end to end: extract, a real njit kernel, write back; and under threads."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import numpy as np
import polars as pl
import pytest
from numba import njit

pytest.importorskip("decider.engine.boundary._arrow._shim", exc_type=ImportError, reason="the Arrow shim can't be built here")

from decider.engine.boundary._arrow.plan import ArrowKindError  # noqa: E402
from decider.engine.boundary.dtypes import NeedsKernelSplit  # noqa: E402
from decider.engine.boundary.extract import extract_frame  # noqa: E402
from decider.engine.boundary.writeback import DtypeGroup, KernelOutputs, Layout, write_back  # noqa: E402
from decider.engine.ir.decls import Input, NullPolicy  # noqa: E402


@njit(cache=False, nogil=True)
def _disposable_income_kernel(net_income, expenses, out):
    for i in range(net_income.shape[0]):
        out[i] = net_income[i] - expenses[i]


def test_extracted_readonly_arrays_feed_a_real_njit_kernel_and_write_back_matches_polars():
    n = 1000
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({"net_income": rng.uniform(2000, 20000, n), "expenses": rng.uniform(500, 6000, n)})
    extracted = extract_frame(frame, [Input("net_income", float), Input("expenses", float)])
    net_income = extracted.columns["net_income"].values
    expenses = extracted.columns["expenses"].values
    assert net_income.flags["WRITEABLE"] is False

    out = np.empty(n, dtype=np.float64)
    _disposable_income_kernel(net_income, expenses, out)
    result = write_back(frame, KernelOutputs(
        float64=DtypeGroup(("disposable_income",), out.reshape(1, n), Layout.COLUMN_MAJOR)))

    expected = (frame["net_income"] - frame["expenses"]).to_numpy()
    assert np.allclose(result["disposable_income"].to_numpy(), expected)
    assert result.columns == ["net_income", "expenses", "disposable_income"]


def _mixed_inputs():
    return [
        Input("balance", float), Input("days", int), Input("active", bool),
        Input("sector", str), Input("bonus", float | None, NullPolicy.OPTIONAL),
        Input("arrears", int, NullPolicy.MISSING_AS, fill=0),
    ]


def _mixed_frame(n=24, seed=5):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "balance": rng.uniform(-1e5, 1e5, n),
        "days": rng.integers(2 ** 53, 2 ** 62, n, dtype=np.int64),
        "active": rng.integers(0, 2, n).astype(bool),
        "sector": rng.choice(["private", "public", "government"], n),
        "bonus": [None if k % 4 == 1 else float(k) for k in range(n)],
        "arrears": [None if k in (0, n - 1) else k for k in range(n)],
    })


def _decoded(ex):
    out = {}
    for name, col in ex.columns.items():
        values = col.values.tolist()
        if col.categories is not None:
            values = [col.categories[c] for c in values]
        if col.validity is not None:
            values = [v if ok else None for v, ok in zip(values, col.validity)]
        out[name] = values
    return out


def test_sixteen_threads_extracting_at_once_agree_with_a_single_threaded_oracle():
    # Pooled row buffers are per thread; a pooling bug shows as one thread's rows in another's result.
    frames = {t: _mixed_frame(n=64 + t, seed=100 + t) for t in range(16)}
    oracles = {t: _decoded(extract_frame(frames[t], _mixed_inputs())) for t in frames}

    def work(t: int) -> int:
        return sum(_decoded(extract_frame(frames[t], _mixed_inputs())) != oracles[t] for _ in range(25))

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert list(pool.map(work, range(16))) == [0] * 16


def test_a_column_the_kernel_cannot_read_fails_by_name():
    with pytest.raises(NeedsKernelSplit, match=r"column 'x' \(List\(Int64\)\)") as exc_info:
        extract_frame(pl.DataFrame({"x": [[1, 2], [3]]}), [Input("x", float)])
    assert isinstance(exc_info.value, ArrowKindError) and exc_info.value.column == "x"
    with pytest.raises(ArrowKindError, match=r"column 'x' \(Struct"):
        extract_frame(pl.DataFrame({"x": [{"a": 1}, {"a": 2}]}), [Input("x", float)])
    with pytest.raises(ArrowKindError, match=r"column 'x' is Arrow string_view but is declared F64"):
        extract_frame(pl.DataFrame({"x": ["not a number"]}), [Input("x", float)])
    ok = extract_frame(pl.DataFrame({"x": pl.Series([Decimal("1.25")], dtype=pl.Decimal(10, 2))}), [Input("x", float)])
    assert ok.columns["x"].values.tolist() == [125.0]
    with pytest.raises(ArrowKindError, match=r"column 'x' \(Decimal\(precision=38, scale=0\)\).*cast to Float64 failed"):
        extract_frame(pl.DataFrame({"x": pl.Series([Decimal("9" * 30)], dtype=pl.Decimal(38, 0))}), [Input("x", float)])


def test_a_nullable_int64_at_100k_rows_stays_int64_with_its_fill_and_no_float_round_trip():
    n = 100_000
    rng = np.random.default_rng(9)
    raw = rng.integers(2 ** 53, 2 ** 62, n, dtype=np.int64)
    nulls = rng.random(n) < 0.1
    frame = pl.DataFrame({"v": pl.Series(np.where(nulls, 0, raw)).set(pl.Series(nulls), None)})
    assert frame["v"].null_count() == int(nulls.sum())
    col = extract_frame(frame, [Input("v", int, NullPolicy.MISSING_AS, fill=-1)]).columns["v"]
    assert col.values.dtype == np.int64
    assert np.array_equal(col.values, np.where(nulls, -1, raw))
