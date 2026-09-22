"""Items 1-4 and 8: ns per row for every matcher over a whole column, on
identical data, at 10k and 1M rows, low and high cardinality.

  pure numba, pointer family (col_ptr)      -- this strand
  pure numba, array family (col_arr)        -- this strand
  pure numba, prefix-only kernel (no kind switch), to price the switch
  C memcmp/memmem via the sibling's shim    -- imported, not rewritten
  PCRE2-JIT via the sibling's shim          -- imported
  polars str.starts_with / ends_with / contains(literal=True) / ==
  per-category: encode + match distinct + lookup per row

Every strategy's output is asserted identical to a Python oracle.
"""
import re, sys, time, pathlib
import numpy as np
import polars as pl
from numba import njit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "numba-c-string-matching"))
import strmatch as SM                      # sibling: C shim (memcmp/memmem + PCRE2-JIT)
from call_ptr import call_match            # sibling: the inttoptr+call intrinsic

import matchers as M
from rawload import load_i64
from data import make_strings, needles
from results_io import record

REPEATS = 7
PYKIND = {"prefix": M.PREFIX, "suffix": M.SUFFIX, "substring": M.SUBSTRING, "substring_absent": M.SUBSTRING, "exact": M.EXACT}
CKIND = {"prefix": SM.PREFIX, "suffix": SM.SUFFIX, "substring": SM.SUBSTRING, "substring_absent": SM.SUBSTRING, "exact": SM.EXACT}


def best(fn, *a, repeats=REPEATS):
    fn(*a); t = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(*a); t.append(time.perf_counter() - t0)
    return min(t)


def py_oracle(kind_name, values, nd):
    nb = nd.encode()
    f = {"prefix": lambda b: b.startswith(nb), "suffix": lambda b: b.endswith(nb),
         "substring": lambda b: nb in b, "substring_absent": lambda b: nb in b, "exact": lambda b: b == nb}[kind_name]
    return np.array([0 if v is None else int(f(v.encode())) for v in values], np.uint8)


@njit(cache=True)
def col_c(fn_table, handle, tab, out):
    """The sibling's C matcher called per row through the pointer table."""
    off_addr = tab[0]; val_addr = tab[1]
    n = np.int64(tab[2]) - 1
    for i in range(n):
        s = load_i64(off_addr + np.uint64(8 * i))
        e = load_i64(off_addr + np.uint64(8 * (i + 1)))
        out[i] = call_match(fn_table[0], handle, val_addr + np.uint64(s), e - s)


@njit(cache=True)
def col_prefix_only(tab, nd_tab, out):
    """Prefix with no kind switch -- prices the switch in col_ptr."""
    off_addr = tab[0]; val_addr = tab[1]
    nd = nd_tab[0]; nlen = np.int64(nd_tab[1])
    n = np.int64(tab[2]) - 1
    for i in range(n):
        s = load_i64(off_addr + np.uint64(8 * i))
        e = load_i64(off_addr + np.uint64(8 * (i + 1)))
        out[i] = M.ptr_prefix(val_addr + np.uint64(s), e - s, nd, nlen)


@njit(cache=True)
def col_lookup(codes, mask, out):
    for i in range(codes.shape[0]):
        out[i] = mask[codes[i]]


def polars_op(kind_name, s, nd):
    if kind_name == "prefix": return s.str.starts_with(nd)
    if kind_name == "suffix": return s.str.ends_with(nd)
    if kind_name in ("substring", "substring_absent"): return s.str.contains(nd, literal=True)
    return s == nd


def main(sizes=(10_000, 1_000_000)):
    fn_table = np.array([SM.MATCH_ADDR], np.uint64)
    for n in sizes:
        for card in ("low", "high"):
            s = make_strings(n, card)
            values = s.to_list()
            t_bufs = best(lambda: M.string_buffers(s))
            offs, vals, valid = M.string_buffers(s)
            tab = M.buffer_table(offs, vals, valid)
            out = np.zeros(n, np.uint8)
            floor = best(M.col_floor, tab, out)
            record("percall", n=n, cardinality=card, matcher="loop floor (offsets load only)", pattern="", kind="",
                   ns_per_row=floor / n * 1e9, get_buffers_ms=t_bufs * 1e3)
            t_enc = best(lambda: s.cast(pl.Categorical))
            cats = s.cast(pl.Categorical)
            codes = cats.to_physical().cast(pl.Int32).to_numpy().copy()
            distinct = cats.cat.get_categories()
            record("percall", n=n, cardinality=card, matcher="category: dictionary-encode column", pattern="", kind="",
                   ns_per_row=t_enc / n * 1e9, n_distinct=len(distinct))
            for kind_name, nd, rx in needles(card, s):
                exp = py_oracle(kind_name, values, nd)
                k = PYKIND[kind_name]
                # --- pure numba, pointer family (kind switch inside) ---
                nt = M.NeedleTable([(k, nd)]); ndt = nt.addr_table(0)
                t = best(M.col_ptr, k, tab, ndt, out); assert (out == exp).all(), ("ptr", kind_name, card)
                record("percall", n=n, cardinality=card, matcher="pure numba ptr", pattern=nd, kind=kind_name,
                       ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9, positives=int(exp.sum()))
                if kind_name.startswith("substring"):
                    t = best(M.col_ptr, M.SUBSTRING_SKIP, tab, ndt, out); assert (out == exp).all()
                    record("percall", n=n, cardinality=card, matcher="pure numba ptr, first-byte skip", pattern=nd, kind=kind_name,
                           ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9)
                if kind_name == "prefix":
                    t = best(M.col_ptr, M.PREFIX_W, tab, ndt, out); assert (out == exp).all()
                    record("percall", n=n, cardinality=card, matcher="pure numba ptr, 8-byte words", pattern=nd, kind=kind_name,
                           ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9)
                    t = best(col_prefix_only, tab, ndt, out); assert (out == exp).all()
                    record("percall", n=n, cardinality=card, matcher="pure numba ptr, prefix-only kernel (no switch)", pattern=nd, kind=kind_name,
                           ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9)
                    # item 8: ASCII case-insensitive prefix
                    ntc = M.NeedleTable([(M.PREFIX_CI, "DoG")])
                    expc = np.array([0 if v is None else int(v.encode().lower().startswith(b"dog")) for v in values], np.uint8)
                    t = best(M.col_ptr, M.PREFIX_CI, tab, ntc.addr_table(0), out); assert (out == expc).all()
                    record("percall", n=n, cardinality=card, matcher="pure numba ptr, ASCII case-insensitive", pattern="DoG", kind="prefix_ci",
                           ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9, positives=int(expc.sum()))
                    tp = best(lambda: s.str.to_lowercase().str.starts_with("dog"))
                    got = s.str.to_lowercase().str.starts_with("dog").fill_null(False).to_numpy().astype(np.uint8); assert (got == expc).all()
                    record("percall", n=n, cardinality=card, matcher="polars to_lowercase + starts_with", pattern="dog", kind="prefix_ci",
                           ns_per_row=tp / n * 1e9)
                # --- pure numba, array family ---
                t = best(M.col_arr, k, offs, vals, valid, nt.buf, nt.start[0], nt.length[0], out); assert (out == exp).all()
                record("percall", n=n, cardinality=card, matcher="pure numba arrays", pattern=nd, kind=kind_name,
                       ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9)
                # --- sibling C literal matcher ---
                h = SM.compile_pattern(nd, CKIND[kind_name], 0)
                t = best(col_c, fn_table, np.int64(h), tab, out); assert (out == exp).all(), ("C", kind_name)
                record("percall", n=n, cardinality=card, matcher="C memcmp/memmem (sibling shim)", pattern=nd, kind=kind_name,
                       ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9)
                SM.free_pattern(h)
                # --- sibling PCRE2-JIT ---
                h = SM.compile_pattern(rx, SM.REGEX, SM.FLAG_UTF)
                t = best(col_c, fn_table, np.int64(h), tab, out); assert (out == exp).all(), ("pcre2", kind_name)
                record("percall", n=n, cardinality=card, matcher="PCRE2-JIT (sibling shim)", pattern=rx, kind=kind_name,
                       ns_per_row=t / n * 1e9, ns_minus_floor=(t - floor) / n * 1e9, jit=SM.is_jit(h))
                SM.free_pattern(h)
                # --- polars whole column ---
                t = best(lambda: polars_op(kind_name, s, nd))
                got = polars_op(kind_name, s, nd).fill_null(False).to_numpy().astype(np.uint8); assert (got == exp).all()
                record("percall", n=n, cardinality=card, matcher="polars whole column", pattern=nd, kind=kind_name, ns_per_row=t / n * 1e9)
                # --- per-category ---
                t_mask = best(lambda: polars_op(kind_name, distinct, nd).fill_null(False).to_numpy().astype(np.uint8))
                mask = polars_op(kind_name, distinct, nd).fill_null(False).to_numpy().astype(np.uint8)
                t_look = best(col_lookup, codes, mask, out); assert (out == exp).all()
                record("percall", n=n, cardinality=card, matcher="per-category: match distinct + lookup (codes already paid)", pattern=nd, kind=kind_name,
                       ns_per_row=(t_mask + t_look) / n * 1e9, mask_ms=t_mask * 1e3, lookup_ms=t_look * 1e3)
                record("percall", n=n, cardinality=card, matcher="per-category incl. encode", pattern=nd, kind=kind_name,
                       ns_per_row=(t_enc + t_mask + t_look) / n * 1e9)


if __name__ == "__main__":
    main()
