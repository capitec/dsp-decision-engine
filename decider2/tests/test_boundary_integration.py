"""End-to-end: the whole-frame Arrow boundary under a real pipeline.

The first two tests are the boundary module's original job (doc 05's scope
line: "getting data out of polars, into a compiled kernel, and back") against
a real `@njit` function on the flagship's own shape. The rest are Stage 3's
acceptance (docs/BOUNDARY-REWORK.md §7, §9, §10 risk 2) through
`Pipeline.apply()` itself: the frame-shape corpus is byte-identical, the
caller's frame is rechunked in place (pinned), every rung of the equivalence
ladder agrees, sixteen threads on one pipeline agree with a single-threaded
oracle, and a column the kernel cannot read fails by name — without touching
tests/test_flagship.py, which is protected.
"""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import numpy as np
import polars as pl
import pytest
from numba import njit

from decider2 import flow, missing_as, param
from decider2._arrow.frame import ArrowKindError
from decider2.boundary.extract import NeedsKernelSplit, extract_frame
from decider2.boundary.writeback import DtypeGroup, KernelOutputs, Layout, write_back
from decider2.testing import assert_equivalent, corpus
from decider2.types import Decision, Input, MissingInputPolicy, NullPolicy


@njit(cache=False, nogil=True)
def _disposable_income_kernel(net_income, expenses, out):
    for i in range(net_income.shape[0]):
        out[i] = net_income[i] - expenses[i]


def test_extracted_readonly_arrays_feed_a_real_njit_kernel_and_write_back_matches_polars():
    n = 1000
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({
        "net_income": rng.uniform(2000, 20000, n),
        "expenses": rng.uniform(500, 6000, n),
    })
    inputs = [
        Input(name="net_income", annotation=float, null_policy=NullPolicy.REQUIRED),
        Input(name="expenses", annotation=float, null_policy=NullPolicy.REQUIRED),
    ]

    extracted = extract_frame(frame, inputs)
    assert extracted.routing.routed_count == 0

    net_income = extracted.columns["net_income"].values
    expenses = extracted.columns["expenses"].values
    assert net_income.flags["WRITEABLE"] is False  # doc 05 §1.4: extraction produces readonly

    out = np.empty(n, dtype=np.float64)  # the kernel's own output buffer — writeable, per §3.1b
    _disposable_income_kernel(net_income, expenses, out)

    outputs = KernelOutputs(
        float64=DtypeGroup(("disposable_income",), out.reshape(1, n), Layout.COLUMN_MAJOR)
    )
    result = write_back(frame, outputs)

    expected = (frame["net_income"] - frame["expenses"]).to_numpy()
    assert np.allclose(result["disposable_income"].to_numpy(), expected)
    assert result.columns == ["net_income", "expenses", "disposable_income"]


def test_a_null_dense_batch_still_produces_a_full_result_via_refer_routing():
    """The production-risk scenario, end to end: some rows have a null in a
    REQUIRED input. The kernel only ever sees the clean subset; the routed
    rows get a Decision instead of blowing up the whole batch (doc 03 §1)."""
    frame = pl.DataFrame({
        "net_income": [9200.0, None, 15000.0, 4999.0],
        "expenses": [3100.0, 1500.0, 6000.0, 2000.0],
    })
    inputs = [
        Input(name="net_income", annotation=float, null_policy=NullPolicy.REQUIRED),
        Input(name="expenses", annotation=float, null_policy=NullPolicy.REQUIRED),
    ]

    extracted = extract_frame(frame, inputs, policy=MissingInputPolicy(default=Decision.REFER, reason=4101))
    assert extracted.routing.routed_count == 1
    assert extracted.routing.column[1] == "net_income"
    assert extracted.kernel_frame.height == 3  # the null row never reaches the kernel

    n = extracted.kernel_frame.height
    out = np.empty(n, dtype=np.float64)
    _disposable_income_kernel(
        extracted.columns["net_income"].values, extracted.columns["expenses"].values, out
    )
    result = write_back(
        extracted.kernel_frame,
        KernelOutputs(float64=DtypeGroup(("disposable_income",), out.reshape(1, n), Layout.COLUMN_MAJOR)),
    )
    assert result["disposable_income"].to_list() == [6100.0, 9000.0, 2999.0]

    routed_rows = frame.filter(pl.Series(extracted.routing.mask))
    assert routed_rows.height == 1
    assert routed_rows["net_income"].to_list() == [None]


# ---------------------------------------------------------------------------
# Stage 3 acceptance through Pipeline.apply()
# ---------------------------------------------------------------------------

def mixed_score(
    balance: float, days: int, active: bool, sector: str,
    bonus: float | None, arrears: int = missing_as(0),
    private: str = param("private"), weight: float = param(0.5),
) -> float:
    """Every kind at once: F64, I64 (above 2**53), BOOL, CODE, OPTIONAL, MISSING_AS."""
    acc = balance * weight + float(days % 1000)
    if active:
        acc += 10.0
    if sector == private:
        acc *= 2.0
    if bonus is not None:
        acc += bonus
    return acc - float(arrears)


def big_days(days: int) -> int:
    """An int64 output computed from an int64 input above 2**53."""
    return days - 1


def numeric_only(balance: float, days: int, active: bool, bonus: float | None, arrears: int = missing_as(0)) -> float:
    """Same shape without the str input, so every rung including score() runs."""
    acc = balance + float(days % 1000)
    if active:
        acc += 10.0
    if bonus is not None:
        acc += bonus
    return acc - float(arrears)


def _mixed_frame(n=24, seed=5):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "balance": rng.uniform(-1e5, 1e5, n),
        "days": rng.integers(2 ** 53, 2 ** 62, n, dtype=np.int64),
        "active": rng.integers(0, 2, n).astype(bool),
        "sector": rng.choice(["private", "public", "government"], n),
        "bonus": [None if k % 4 == 1 else float(k) for k in range(n)],
        "arrears": [None if k in (0, n - 1) else k for k in range(n)],       # the null on row 0 AND the last row
    })


def _same(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert a.columns == b.columns and a.dtypes == b.dtypes
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.is_float():
            assert np.array_equal(x.to_numpy(), y.to_numpy(), equal_nan=True), c
        else:
            assert x.to_list() == y.to_list(), c


@pytest.fixture(scope="module")
def mixed_pipeline():
    return flow(mixed_score, big_days)


def test_the_frame_shape_corpus_through_apply_is_byte_identical(mixed_pipeline):
    """§8.2 item 3: a fresh frame, `df[a:b]`, a `pl.concat` multi-chunk frame
    and a nullable numeric with the null on row 0 and on the last row all
    give the same answer — in every mode, since every mode is the same
    Arrow import now (§1.6)."""
    fresh = _mixed_frame()
    for mode in ("fused", "stepped", "interpreted"):
        reference = mixed_pipeline.apply(fresh, mode=mode)
        assert reference["mixed_score"].dtype == pl.Float64 and reference["big_days"].dtype == pl.Int64
        assert reference["big_days"].to_list() == [d - 1 for d in fresh["days"].to_list()]     # exact above 2**53

        sliced = _mixed_frame()[5:19]
        _same(mixed_pipeline.apply(sliced, mode=mode), reference[5:19])

        multi = pl.concat([_mixed_frame()[0:7], _mixed_frame()[7:16], _mixed_frame()[16:24]])
        assert multi.n_chunks() == 3
        _same(mixed_pipeline.apply(multi, mode=mode), reference)

        for a, b in ((0, 1), (23, 24), (0, 24)):
            _same(mixed_pipeline.apply(_mixed_frame()[a:b], mode=mode), reference[a:b])


def test_the_callers_frame_is_rechunked_in_place_by_apply(mixed_pipeline):
    """§9 (Stage 3): polars rechunks the frame IN PLACE while exporting it.
    Harmless — it is what `rechunk_once` did to a copy before — but a
    caller holding numpy views into a multi-chunk frame taken BEFORE
    `apply()` now holds views onto the old chunks. Pinned so a polars
    change is noticed. Measured rule (polars 1.41.2): the export rechunks
    whichever frame it is called on, aligned or not; a `clone()` has its
    own column vector, so when a frame-tier cast or a routing filter put a
    derived frame in front of the export, the CALLER's frame is untouched."""
    numeric = flow(numeric_only)                                        # no cast: the caller's frame is exported
    multi = pl.concat([_mixed_frame()[0:7], _mixed_frame()[7:24]]).drop("sector")
    assert multi.n_chunks("all") == [2] * multi.width
    numeric.apply(multi)
    assert multi.n_chunks("all") == [1] * multi.width

    multi = pl.concat([_mixed_frame()[0:7], _mixed_frame()[7:24]])      # `sector: str` -> a frame-tier cast -> a clone
    assert multi.n_chunks("all") == [2] * multi.width
    mixed_pipeline.apply(multi)
    assert multi.n_chunks("all") == [2] * multi.width


def test_every_rung_of_the_equivalence_ladder_agrees_on_the_boundary_corpus(mixed_pipeline):
    """`assert_equivalent` drives interpreted / stepped / fused and then
    `score()` per row (doc 02 §3.1; §1.6: the fourth rung is the
    independent, Python-native producer). Numeric-only so every rung runs;
    the boundary corpus has nulls in REQUIRED, OPTIONAL and MISSING_AS
    columns and an int64 past 2**53."""
    numeric = flow(numeric_only, big_days)
    frames = corpus(numeric)
    assert_equivalent(numeric, frames["boundary"])
    assert_equivalent(numeric, frames["empty"])
    assert_equivalent(numeric, _mixed_frame().drop("sector"))
    assert_equivalent(numeric, pl.concat([_mixed_frame()[0:9], _mixed_frame()[9:24]]).drop("sector"))
    # the mixed one too: three rungs (score() skips str inputs until Stage 2 lands)
    assert_equivalent(mixed_pipeline, corpus(mixed_pipeline)["boundary"])
    assert_equivalent(mixed_pipeline, _mixed_frame()[3:21])


def test_sixteen_threads_on_one_pipeline_agree_with_a_single_threaded_oracle(mixed_pipeline):
    """§10 risk 2: pooled ctypes structs and row buffers are per THREAD. A
    pooled-buffer bug shows as cross-talk — one thread's row answered with
    another's — so every thread gets its own distinct frame and record and
    checks every result against the oracle computed before any thread
    started."""
    frames = {t: _mixed_frame(n=64 + t, seed=100 + t) for t in range(16)}
    oracles = {t: mixed_pipeline.apply(frames[t]) for t in frames}
    records = {t: frames[t].row(t % 3, named=True) for t in frames}
    score_oracles = {t: mixed_pipeline.score(records[t]) for t in frames}

    def work(t: int) -> int:
        bad = 0
        for _ in range(25):
            out = mixed_pipeline.apply(frames[t])
            if not (out["mixed_score"].to_list() == oracles[t]["mixed_score"].to_list()
                    and out["big_days"].to_list() == oracles[t]["big_days"].to_list()):
                bad += 1
            if mixed_pipeline.score(records[t]) != score_oracles[t]:
                bad += 1
        return bad

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(16)))
    assert results == [0] * 16, results


def test_a_column_the_kernel_cannot_read_fails_by_name_through_apply():
    def uses_x(x: float) -> float:
        """Reads one column."""
        return x

    p = flow(uses_x)
    with pytest.raises(NeedsKernelSplit, match=r"column 'x' \(List\(Int64\)\)") as exc_info:
        p.apply(pl.DataFrame({"x": [[1, 2], [3]]}))
    assert isinstance(exc_info.value, ArrowKindError) and exc_info.value.column == "x"
    with pytest.raises(ArrowKindError, match=r"column 'x' \(Struct"):
        p.apply(pl.DataFrame({"x": [{"a": 1}, {"a": 2}]}))
    with pytest.raises(ArrowKindError, match=r"column 'x' is Arrow string_view but is declared F64"):
        p.apply(pl.DataFrame({"x": ["not a number"]}))
    # Decimal is the frame-tier cast case (doc 03 §1.2): cents on the wire, as before ...
    out = p.apply(pl.DataFrame({"x": pl.Series([Decimal("1.25")], dtype=pl.Decimal(10, 2))}))
    assert out["uses_x"].to_list() == [125.0]
    # ... and a Decimal that cannot be cast fails by name, not by crash.
    with pytest.raises(ArrowKindError, match=r"column 'x' \(Decimal\(precision=38, scale=0\)\).*cast to Float64 failed"):
        p.apply(pl.DataFrame({"x": pl.Series([Decimal("9" * 30)], dtype=pl.Decimal(38, 0))}))


def test_a_nullable_int64_at_100k_rows_stays_int64_with_its_fill_and_no_float_round_trip():
    """The nullable-numeric case that used to copy through `np.where`
    (doc 05 §1.5, ~550-630 µs/100k rows) and, declared `int`, round-trip
    through float64 in the old `astype`: it is one gather now, and the
    values above 2**53 come back exact."""
    n = 100_000
    rng = np.random.default_rng(9)
    raw = rng.integers(2 ** 53, 2 ** 62, n, dtype=np.int64)
    nulls = rng.random(n) < 0.1
    frame = pl.DataFrame({"v": pl.Series(np.where(nulls, 0, raw)).set(pl.Series(nulls), None)})
    assert frame["v"].null_count() == int(nulls.sum())
    col = extract_frame(frame, [Input("v", int, NullPolicy.MISSING_AS, fill=-1)]).columns["v"]
    assert col.values.dtype == np.int64 and col.fill.filled_count == int(nulls.sum())
    expected = np.where(nulls, -1, raw)
    assert np.array_equal(col.values, expected)
