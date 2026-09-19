"""End-to-end: extract -> a real numba kernel -> write_back.

This is the boundary module's actual job (doc 05's scope line: "getting data
out of polars, into a compiled kernel, and back") exercised against a real
`@njit` function, on the flagship pipeline's own shape (doc 03 §1.1) — without
touching tests/test_flagship.py, which is the acceptance target for the
`graph`/`compile` layers this module feeds, not for `boundary` alone.

Kept at 1k rows per the task's memory discipline; this is a correctness
check, not a benchmark (EXPERIMENTS.md already carries the batch numbers).
"""
import numpy as np
import polars as pl
from numba import njit

from decider2.boundary.extract import extract_frame
from decider2.boundary.writeback import DtypeGroup, KernelOutputs, Layout, write_back
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

    # The routed row is recovered from the ORIGINAL frame by row position,
    # via the routing mask — this is what a runtime layer does to merge a
    # Decision back in for the row the kernel never saw.
    routed_rows = frame.filter(pl.Series(extracted.routing.mask))
    assert routed_rows.height == 1
    assert routed_rows["net_income"].to_list() == [None]
