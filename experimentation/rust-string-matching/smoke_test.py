"""Correctness smoke test: all strategies agree with Python `re` on both
cardinalities, and `match_rows` serves column, dictionary, and
dictionary+codes through the one function."""
from __future__ import annotations
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, polars as pl
from binding import (StringBuffers, compile_pattern, make_ptr_table, match_rows_py, match_one_py)
from data import make_data
import kernels

PT = make_ptr_table()
pid = compile_pattern("^dog")
for card in ("low", "high"):
    df, employers = make_data(20_000, card, seed=3)
    s = df["ft3"]
    sb = StringBuffers(s)
    ref = np.array([bool(re.match("^dog", x)) for x in s.to_list()])
    # column mode
    m_col = match_rows_py(pid, sb)
    assert np.array_equal(m_col.astype(bool), ref), card
    # dictionary mode + codes
    cat = s.cast(pl.Categorical)
    codes = cat.to_physical().to_numpy().astype(np.int32)
    cats = cat.cat.get_categories()
    dsb = StringBuffers(cats)
    m_dict = match_rows_py(pid, dsb)
    assert np.array_equal(m_dict[codes].astype(bool), ref), card
    m_dc = match_rows_py(pid, dsb, codes)
    assert np.array_equal(m_dc.astype(bool), ref), card
    # match_one
    assert all(match_one_py(pid, sb, i) == int(ref[i]) for i in range(0, 20_000, 997))
    # kernels at 50% selectivity
    ft1, ft2 = df["ft1"].to_numpy(), df["ft2"].to_numpy()
    c1, c2 = 0.5, 0.0
    exp = (ft1 > c1) & (ft2 > c2) & ref
    o = np.empty(len(ft1), dtype=np.bool_)
    fm = s.str.contains("^dog").to_numpy()
    kernels.frame_tier(ft1, ft2, fm, c1, c2, o); assert np.array_equal(o, exp)
    kernels.per_category(ft1, ft2, codes, m_dict, c1, c2, o); assert np.array_equal(o, exp)
    kernels.lazy_rust(ft1, ft2, sb.offsets, sb.values, PT, np.int32(pid), c1, c2, o); assert np.array_equal(o, exp)
    kernels.lazy_rust_via_dict(ft1, ft2, codes, dsb.offsets, dsb.values, PT, np.int32(pid), c1, c2, o); assert np.array_equal(o, exp)
    print(card, "ok: distinct =", len(cats), "matches =", int(ref.sum()), "hits@50% =", int(exp.sum()))
for kind, pat, pyfn in [("starts_with", "dog", lambda x: x.startswith("dog")), ("contains", "og", lambda x: "og" in x),
                        ("ends_with", "inc", lambda x: x.endswith("inc")), ("exact", "vodacom", lambda x: x == "vodacom"),
                        ("regex", "s$", lambda x: bool(re.search("s$", x)))]:
    k = compile_pattern(pat, kind); got = match_rows_py(k, sb).astype(bool)
    assert np.array_equal(got, np.array([pyfn(x) for x in s.to_list()])), kind
print("all five match kinds agree with Python on the high-card column")
print("bad id ->", match_one_py(999, sb, 0), " oob idx ->", match_one_py(pid, sb, 10**9))
print("all strategies agree")
