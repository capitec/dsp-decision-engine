"""THE measurement: the owner's shape `ft1 > c1 AND ft2 > c2 AND ft3 ~ /^dog/`
at 0% / 1% / 50% / 100% of rows reaching the regex node, at low (12
distinct) and high (~1 per row) cardinality, three strategies:

  frame     -- polars `str.contains` over EVERY row before the kernel
               (today's route; the string node becomes IS_TRUE on a bool),
  category  -- dictionary-encode, run the regex over the DISTINCT values,
               kernel does mask[code] (§T's proposal),
  lazy      -- typed arrays + C called at the node (this experiment).

Every strategy's per-batch cost is timed end to end (its own precompute +
the walk), identical outputs asserted, and a Python `re` oracle checks the
frame result once per (pattern, cardinality).
"""
import re, sys, time
import numpy as np
import polars as pl
import strmatch as SM
import kernels as K
from kernels import *
from results_io import record

N = 1_000_000
REPEATS = 5
HARD = r"^d[o0]g\w*(ma|e|s)$"
TYPED_KEYS = ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")

def best(fn, repeats=REPEATS):
    fn(); t = []
    for _ in range(repeats):
        t0 = time.perf_counter(); r = fn(); t.append(time.perf_counter() - t0)
    return min(t), r

def make_strings(n, cardinality, rng):
    base = ["dog", "doge", "cat", "dogma", "hotdog", "bird", "fish", "dogs", "cow", "dodo", "dig", "dögg"]
    if cardinality == "low":
        return pl.Series("ft3", np.array(base)[rng.integers(0, 12, n)])
    return pl.Series("ft3", [f"{base[i % 12]}-{i:07d}" for i in range(n)])

def run_typed(inp, prog, out):
    K.walk_typed(*[inp[k] for k in TYPED_KEYS], *prog.arrays(), out)
    return out

def main():
    rng = np.random.default_rng(7)
    ft1 = rng.uniform(0, 10, N); ft2 = rng.uniform(0, 20, N)
    f64s = np.column_stack([ft1, ft2])
    fn_table = np.array([SM.MATCH_ADDR], np.uint64)
    for card in ("low", "high"):
        s = make_strings(N, card, rng)
        for pat in ("^dog", HARD):
            # oracle for the string test itself, once
            pre_bool = s.str.contains(pat).to_numpy()
            ora = np.fromiter((bool(re.search(pat, x)) for x in s.to_list()), bool, N)
            assert (pre_bool == ora).all()
            handle = SM.compile_pattern(pat)
            # --- per-strategy precompute, timed on its own -------------------
            t_frame_pre, pre_u8 = best(lambda: s.str.contains(pat).to_numpy().astype(np.uint8))
            t_encode, (codes, n_cat) = best(lambda: (lambda c: (c.to_physical().cast(pl.Int32).to_numpy().reshape(-1, 1).copy(), c.cat.get_categories()))(s.cast(pl.Categorical)))
            cats = s.cast(pl.Categorical)
            t_mask, mask = best(lambda: cats.cat.get_categories().str.contains(pat).to_numpy().astype(np.uint8)[None, :].copy())
            t_bufs, (offs, vals) = best(lambda: K.string_buffers(s))
            n_distinct = int(s.n_unique())
            record("selectivity_precompute", cardinality=card, n_distinct=n_distinct, pattern=pat, n=N,
                   frame_contains_ms=t_frame_pre*1e3, category_encode_ms=t_encode*1e3, category_mask_ms=t_mask*1e3,
                   lazy_get_buffers_ms=t_bufs*1e3)
            for sel in (0.0, 0.01, 0.5, 1.0):
                c1 = 11.0 if sel == 0 else float(np.quantile(ft1, 1 - sel))
                c2 = -1.0   # ft2 > c2 always true: the second AND term is walked but never filters
                thr = np.array([c1, c2])
                out = np.zeros(N, np.int32)
                # A. today: one float64 array, string test hoisted to float64 bool
                feats = np.column_stack([ft1, ft2, pre_u8.astype(np.float64)])
                progA = K.owner_shape((IS_TRUE, F64, 2, 0, 0, 4, 3, 0))
                t_walk_f64, _ = best(lambda: K.walk_f64(feats, thr, *progA.arrays_f64(), out))
                outA = out.copy()
                # A'. frame precompute, typed walk with IS_TRUE on the uint8 column
                inp = K.empty_typed_inputs(N); inp.update(f64s=f64s, thr_f64=thr, u8s=pre_u8.reshape(-1, 1).copy(), fn_table=fn_table)
                progF = K.owner_shape((IS_TRUE, U8, 0, 0, 0, 4, 3, 0)).validate(2, 0, 1, 0, 0, 2, 0, 0, 0)
                t_walk_frame, _ = best(lambda: run_typed(inp, progF, out)); outF = out.copy()
                # B. per-category: mask[code]
                inp = K.empty_typed_inputs(N); inp.update(f64s=f64s, thr_f64=thr, i32s=codes, masks=mask, fn_table=fn_table)
                progC = K.owner_shape((STR_MASK, I32, 0, 0, 0, 4, 3, 0)).validate(2, 0, 0, 1, 0, 2, 0, 0, 1)
                t_walk_cat, _ = best(lambda: run_typed(inp, progC, out)); outC = out.copy()
                # C. lazy: C at the node
                inp = K.empty_typed_inputs(N); inp.update(f64s=f64s, thr_f64=thr, str_tab=K.string_table([(offs, vals)]),
                                                          handles=np.array([handle], np.int64), fn_table=fn_table)
                progL = K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0)).validate(2, 0, 0, 0, 1, 2, 0, 1, 0)
                t_walk_lazy, _ = best(lambda: run_typed(inp, progL, out)); outL = out.copy()
                expected = ((ft1 > c1) & (ft2 > c2) & ora).astype(np.int32)
                for name, o in (("f64", outA), ("frame", outF), ("category", outC), ("lazy", outL)):
                    assert (o == expected).all(), (name, card, pat, sel)
                reached = int(((ft1 > c1) & (ft2 > c2)).sum())
                record("selectivity", cardinality=card, n_distinct=n_distinct, pattern=pat, n=N, selectivity=sel,
                       rows_reaching_regex=reached, positives=int(expected.sum()),
                       today_f64_walk_ms=t_walk_f64*1e3, today_total_ms=(t_frame_pre + t_walk_f64)*1e3,
                       frame_walk_ms=t_walk_frame*1e3, frame_total_ms=(t_frame_pre + t_walk_frame)*1e3,
                       category_walk_ms=t_walk_cat*1e3, category_total_excl_encode_ms=(t_mask + t_walk_cat)*1e3,
                       category_total_incl_encode_ms=(t_encode + t_mask + t_walk_cat)*1e3,
                       lazy_walk_ms=t_walk_lazy*1e3, lazy_total_ms=(t_bufs + t_walk_lazy)*1e3,
                       identical=True)
            SM.free_pattern(handle)

if __name__ == "__main__":
    main()
