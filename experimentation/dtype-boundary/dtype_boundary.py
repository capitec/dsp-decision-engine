"""
EXPERIMENT A - what dtypes can actually cross the polars -> numba boundary.

Tests doc 05 section 1 ("Extraction") of decider2/docs/05-boundary-and-compilation.md
against the real environment. For each polars dtype, in a clean and a null-bearing
form, it measures:

  1. buffer layout   - does the documented 2-tuple Arrow unpack work? how many
                       buffers are there really?
  2. to_numpy()      - resulting dtype, copy or not, lossy or not
  3. njit            - can a trivial compiled kernel consume the extraction at all,
                       and if not, what is the exact numba error
  4. extraction cost - per column at 100k rows, median of repeats

Plus the five specific traps doc 05 section 1 asserts.

Run:  <repo>/.venv/bin/python experimentation/dtype-boundary/dtype_boundary.py
Add --quick to skip the timing sweep.
"""

from __future__ import annotations

import datetime as dt
import decimal
import statistics
import sys
import time
import traceback

import numpy as np
import polars as pl
from numba import njit

N_TIME = 100_000
REPEATS = 15


def short_exc(e: BaseException, limit: int = 160) -> str:
    msg = " ".join(str(e).split())
    if len(msg) > limit:
        msg = msg[:limit] + "..."
    return f"{type(e).__name__}: {msg}"


# --------------------------------------------------------------------------
# dtype catalogue.  Each entry builds a clean and a null-bearing series of n rows.
# --------------------------------------------------------------------------

def _cyc(vals, n):
    return [vals[i % len(vals)] for i in range(n)]


def _nullify(vals):
    out = list(vals)
    for i in range(3, len(out), 7):
        out[i] = None
    return out


DTYPES: list[tuple[str, object, object]] = []


def reg(name, dtype, gen):
    DTYPES.append((name, dtype, gen))


reg("Float64", pl.Float64, lambda n: _cyc([1.5, 2.5, 3.25], n))
reg("Int64", pl.Int64, lambda n: _cyc([1, 2, 3], n))
reg("Int32", pl.Int32, lambda n: _cyc([1, 2, 3], n))
reg("UInt8", pl.UInt8, lambda n: _cyc([1, 2, 250], n))
reg("Boolean", pl.Boolean, lambda n: _cyc([True, False, True], n))
reg("Utf8", pl.Utf8, lambda n: _cyc(["PERS_LOAN", "CARD", "HOMELOAN"], n))
reg("Categorical", pl.Categorical, lambda n: _cyc(["PRIVATE", "PUBLIC", "SME"], n))
reg("Enum", pl.Enum(["A", "B", "C"]), lambda n: _cyc(["A", "B", "C"], n))
reg("Date", pl.Date, lambda n: _cyc([dt.date(2020, 1, 1), dt.date(2021, 6, 30), dt.date(2024, 2, 29)], n))
reg("Datetime(us)", pl.Datetime("us"), lambda n: _cyc(
    [dt.datetime(2020, 1, 1, 12), dt.datetime(2021, 6, 30, 3, 15), dt.datetime(2024, 2, 29, 23, 59)], n))
reg("Duration(us)", pl.Duration("us"), lambda n: _cyc(
    [dt.timedelta(days=30), dt.timedelta(days=90), dt.timedelta(hours=5)], n))
reg("Decimal(18,2)", pl.Decimal(18, 2), lambda n: _cyc(
    [decimal.Decimal("1234.56"), decimal.Decimal("0.01"), decimal.Decimal("999999.99")], n))
reg("List(Float64)", pl.List(pl.Float64), lambda n: _cyc([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], n))


def make_series(dtype, gen, n, nullable):
    vals = gen(n)
    if nullable:
        vals = _nullify(vals)
    return pl.Series("c", vals, dtype=dtype)


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def probe_to_arrow(s: pl.Series) -> dict:
    """Doc 05 1.2 verbatim: a = series.to_arrow(); v, d = a.buffers()."""
    out = {}
    try:
        a = s.to_arrow()
    except BaseException as e:
        out["to_arrow"] = short_exc(e)
        out["n_buffers"] = None
        out["two_tuple_unpack"] = "n/a (to_arrow failed)"
        return out
    out["to_arrow"] = "ok"
    try:
        bufs = a.buffers()
        out["n_buffers"] = len(bufs)
    except BaseException as e:
        out["n_buffers"] = short_exc(e)
    try:
        validity_buf, values_buf = a.buffers()
        out["two_tuple_unpack"] = "ok"
    except BaseException as e:
        out["two_tuple_unpack"] = short_exc(e)
    return out


def probe_native_buffers(s: pl.Series) -> dict:
    """polars-native equivalent of the Arrow unpack (no pyarrow needed)."""
    out = {}
    try:
        b = s._get_buffers()
    except BaseException as e:
        out["_get_buffers"] = short_exc(e)
        return out
    present = [k for k, v in b.items() if v is not None]
    out["_get_buffers"] = "ok"
    out["native_parts"] = ",".join(present) if present else "(none)"
    out["n_native_parts"] = len(present)
    v = b.get("validity")
    out["validity_is_none"] = v is None
    vals = b.get("values")
    out["values_dtype"] = str(vals.dtype) if vals is not None else None
    try:
        out["buffer_info"] = s._get_buffer_info()
    except BaseException as e:
        out["buffer_info"] = short_exc(e)
    return out


def probe_to_numpy(s: pl.Series) -> dict:
    out = {}
    try:
        a = s.to_numpy()
        out["np_dtype"] = str(a.dtype)
        out["np_writeable"] = bool(a.flags.writeable)
    except BaseException as e:
        out["np_dtype"] = short_exc(e)
        out["np_writeable"] = None
        a = None
    # zero copy?
    try:
        z = s.to_numpy(allow_copy=False)
        out["allow_copy_False"] = "ok"
        try:
            ptr = s._get_buffer_info()[0]
            out["shares_address"] = (z.__array_interface__["data"][0] == ptr)
        except BaseException:
            out["shares_address"] = None
    except BaseException as e:
        out["allow_copy_False"] = short_exc(e, 90)
        out["shares_address"] = False
    return out


# --- njit kernels ---------------------------------------------------------

@njit(cache=False)
def k_touch(a):
    n = a.shape[0]
    acc = 0
    for i in range(n):
        if a[i] == a[0]:
            acc += 1
    return acc


@njit(cache=False)
def k_arith(a):
    s = a[0]
    for i in range(1, a.shape[0]):
        s = s + a[i]
    return s


def probe_njit(arr) -> dict:
    out = {}
    if not isinstance(arr, np.ndarray):
        return {"njit_touch": "no array to pass", "njit_arith": "-"}
    for label, fn in (("njit_touch", k_touch), ("njit_arith", k_arith)):
        try:
            fn(arr)
            out[label] = "ok"
        except BaseException as e:
            out[label] = short_exc(e, 190)
    return out


def median_us(fn, repeats=REPEATS):
    fn()  # warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


# --------------------------------------------------------------------------
# the five documented traps
# --------------------------------------------------------------------------

def trap_report():
    lines = []

    def say(name, claim, verdict, detail):
        lines.append(f"\n[{verdict}] {name}\n  claim : {claim}\n  result: {detail}")

    # TRAP 1 - rechunk() does not reset offset
    base = pl.Series("x", [float(i) for i in range(20)])
    sl = base.slice(9, 3)
    try:
        pi_before = sl._get_buffer_info()
        rc = sl.rechunk()
        pi_after = rc._get_buffer_info()
        base_ptr = base._get_buffer_info()[0]
        detail = (f"base ptr={base_ptr}; slice(9,3) buffer_info={pi_before} "
                  f"(ptr-base={pi_before[0]-base_ptr} bytes, offset field={pi_before[1]}); "
                  f"after rechunk={pi_after} (ptr-base={pi_after[0]-base_ptr}); "
                  f"values={list(rc)}")
        # in polars-native terms the offset field stayed 0 and the POINTER moved
        verdict = "DIFFERENT MECHANISM" if pi_before[1] == 0 else "REPRODUCES"
    except BaseException as e:
        detail = short_exc(e)
        verdict = "COULD NOT RUN"
    say("TRAP 1 rechunk does not reset a.offset",
        "'rechunk() does not reset a.offset. A slice(9,12) still reports offset=9'",
        verdict, detail)

    # TRAP 2 - validity bit i maps to element offset+i
    vals = [float(i) for i in range(20)]
    vals[10] = None
    s2 = pl.Series("y", vals)
    sl2 = s2.slice(9, 3)
    try:
        full_v = s2._get_buffers()["validity"]
        sl_v = sl2._get_buffers()["validity"]
        detail = (f"full validity (len {len(full_v)}) around idx 9..11 = {list(full_v)[9:12]}; "
                  f"slice(9,3) validity = {list(sl_v) if sl_v is not None else None} "
                  f"(len {len(sl_v) if sl_v is not None else 0}); "
                  f"slice values = {list(sl2)}")
        verdict = "DOES NOT REPRODUCE (polars pre-slices validity)" if (
            sl_v is not None and len(sl_v) == 3) else "REPRODUCES"
    except BaseException as e:
        detail = short_exc(e)
        verdict = "COULD NOT RUN"
    say("TRAP 2 validity must be unpacked before slicing",
        "'Validity must be unpacked before slicing, because bit i maps to element offset + i'",
        verdict, detail)

    # TRAP 3 - clean column has validity_buf is None
    res = []
    for name, dtype, gen in DTYPES:
        for nullable in (False, True):
            try:
                s = make_series(dtype, gen, 64, nullable)
                b = s._get_buffers()
                res.append(f"{name}{'/null' if nullable else '/clean'}="
                           f"{'None' if b.get('validity') is None else 'present'}")
            except BaseException as e:
                res.append(f"{name}{'/null' if nullable else '/clean'}=ERR({type(e).__name__})")
    say("TRAP 3 clean column has validity_buf is None",
        "'A column with no nulls has validity_buf is None - no bitmap is allocated at all'",
        "SEE DETAIL", "; ".join(res))

    # TRAP 4 - bool not zero copy
    sb = pl.Series("b", [True, False] * 50_000)
    try:
        sb.to_numpy(allow_copy=False)
        d4, v4 = "allow_copy=False SUCCEEDED (no raise)", "REFUTED"
    except BaseException as e:
        d4, v4 = f"allow_copy=False -> {short_exc(e, 120)}", "REPRODUCES"
    try:
        cost = median_us(lambda: sb.to_numpy())
        d4 += f"; to_numpy() cost at 100k rows = {cost:.1f} us"
    except BaseException as e:
        d4 += f"; cost: {short_exc(e)}"
    say("TRAP 4 bool columns are not zero-copy", "'Arrow bitpacks them; allow_copy=False raises. "
        "Budget ~48 us/col at 100k rows'", v4, d4)

    # TRAP 5 - the value slot under a null holds garbage, not zero
    left = pl.DataFrame({"k": [1, 2, 3, 4, 5]})
    right = pl.DataFrame({"k": [1, 3, 5], "v": [9.0, 8.0, 7.0]})
    j = left.join(right, on="k", how="left")
    findings = []
    try:
        vb = j["v"]._get_buffers()
        raw = vb["values"].to_numpy()
        findings.append(f"left-join values buffer = {list(raw)} (nulls at {[i for i,x in enumerate(list(j['v'])) if x is None]})")
        nonzero_under_null = any(raw[i] != 0.0 for i, x in enumerate(list(j["v"])) if x is None)
        findings.append(f"any non-zero under a null: {nonzero_under_null}")
    except BaseException as e:
        findings.append(short_exc(e))
        nonzero_under_null = None
    # a literal-built nullable series, for contrast
    try:
        lit = pl.Series("l", [1.0, None, 3.0])
        findings.append(f"literal-built nullable values buffer = {list(lit._get_buffers()['values'])}")
    except BaseException as e:
        findings.append(short_exc(e))
    # to_numpy of a nullable int
    try:
        big = pl.Series("i", [2**60 + 1, None, 3], dtype=pl.Int64)
        a = big.to_numpy()
        findings.append(f"Int64+null .to_numpy() -> dtype={a.dtype}, first={a[0]!r} "
                        f"(exact={a[0] == 2**60 + 1 if a.dtype.kind=='i' else 'N/A float'})")
    except BaseException as e:
        findings.append(short_exc(e))
    v5 = ("REPRODUCES" if nonzero_under_null else
          "DOES NOT REPRODUCE on this join" if nonzero_under_null is False else "COULD NOT RUN")
    say("TRAP 5 value slot under a null holds garbage", "'The slot under a null contains leftover "
        "garbage, not zero. A measured left join produced 9.0 at a position that was genuinely null'",
        v5, " | ".join(findings))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    quick = "--quick" in sys.argv
    print("=" * 110)
    print("EXPERIMENT A - polars -> numba boundary, dtype by dtype")
    print(f"python {sys.version.split()[0]}  polars {pl.__version__}  numpy {np.__version__}")
    import numba
    print(f"numba {numba.__version__}")
    try:
        import pyarrow
        print(f"pyarrow {pyarrow.__version__}")
    except BaseException as e:
        print(f"pyarrow ABSENT -> {short_exc(e)}")
    print("=" * 110)

    rows = []
    for name, dtype, gen in DTYPES:
        for nullable in (False, True):
            label = f"{name}{' +null' if nullable else ''}"
            rec = {"dtype": label}
            try:
                s = make_series(dtype, gen, 1000, nullable)
            except BaseException as e:
                rec["build"] = short_exc(e)
                rows.append(rec)
                continue
            rec.update(probe_to_arrow(s))
            rec.update(probe_native_buffers(s))
            rec.update(probe_to_numpy(s))
            arr = None
            try:
                arr = s.to_numpy()
            except BaseException:
                arr = None
            rec.update(probe_njit(arr))
            # njit on the raw values buffer (nulls stripped out of the picture)
            try:
                vb = s._get_buffers()["values"]
                vraw = vb.to_numpy()
                r2 = probe_njit(vraw)
                rec["njit_valuesbuf_touch"] = r2["njit_touch"]
                rec["raw_values_np_dtype"] = str(vraw.dtype)
            except BaseException as e:
                rec["njit_valuesbuf_touch"] = short_exc(e, 120)
                rec["raw_values_np_dtype"] = "-"
            rows.append(rec)

    # ---- table 1: buffers + numpy ----
    print("\n### TABLE 1 - buffer layout and numpy extraction\n")
    hdr = f"{'dtype':<16}{'to_arrow':<24}{'native parts':<26}{'valid=None':<11}{'to_numpy dtype':<18}{'zero-copy':<10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dtype']:<16}{str(r.get('to_arrow','-'))[:23]:<24}"
              f"{str(r.get('native_parts','-'))[:25]:<26}"
              f"{str(r.get('validity_is_none','-')):<11}"
              f"{str(r.get('np_dtype','-'))[:17]:<18}"
              f"{('YES' if r.get('allow_copy_False')=='ok' else 'no'):<10}")

    # ---- table 2: njit ----
    print("\n### TABLE 2 - can an njit kernel consume it?\n")
    for r in rows:
        print(f"{r['dtype']:<16} to_numpy[{r.get('np_dtype','-')}] touch={r.get('njit_touch','-')}")
        print(f"{'':<16} arith={r.get('njit_arith','-')}")
        print(f"{'':<16} values-buffer[{r.get('raw_values_np_dtype','-')}] touch={r.get('njit_valuesbuf_touch','-')}")

    # ---- table 3: extraction cost ----
    if not quick:
        print(f"\n### TABLE 3 - extraction cost per column, {N_TIME} rows, median of {REPEATS}\n")
        print(f"{'dtype':<16}{'to_numpy us':>14}{'no-copy us':>14}{'_get_buffers us':>18}{'zero-copy':>12}")
        print("-" * 74)
        for name, dtype, gen in DTYPES:
            for nullable in (False, True):
                label = f"{name}{' +null' if nullable else ''}"
                try:
                    s = make_series(dtype, gen, N_TIME, nullable)
                except BaseException as e:
                    print(f"{label:<16} build failed: {short_exc(e, 40)}")
                    continue
                try:
                    t_copy = median_us(lambda s=s: s.to_numpy())
                    t_copy_s = f"{t_copy:>14.1f}"
                except BaseException:
                    t_copy_s = f"{'raise':>14}"
                try:
                    s.to_numpy(allow_copy=False)
                    t_nc = median_us(lambda s=s: s.to_numpy(allow_copy=False))
                    t_nc_s, zc = f"{t_nc:>14.2f}", "YES"
                except BaseException:
                    t_nc_s, zc = f"{'raise':>14}", "no"
                try:
                    t_gb = median_us(lambda s=s: s._get_buffers())
                    t_gb_s = f"{t_gb:>18.2f}"
                except BaseException:
                    t_gb_s = f"{'raise':>18}"
                print(f"{label:<16}{t_copy_s}{t_nc_s}{t_gb_s}{zc:>12}")

    print("\n\n### DOCUMENTED TRAPS")
    print(trap_report())


if __name__ == "__main__":
    main()
