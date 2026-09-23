"""Does the whole thing work at all? Both walkers, all three string
strategies, same answer as a Python oracle."""
import re
import numpy as np
import polars as pl
import kernels as K
import strmatch as SM
from kernels import *

n = 20_000
rng = np.random.default_rng(0)
ft1 = rng.uniform(0, 10, n); ft2 = rng.uniform(0, 20, n)
words = np.array(["dog", "doge", "cat", "dogma", "hotdog", "bird", "", "dög", "DOG", "d"])
ft3 = pl.Series("ft3", words[rng.integers(0, len(words), n)])
pat = r"^dog"
oracle = np.array([(a > 5) and (b > 10) and bool(re.search(pat, s)) for a, b, s in zip(ft1, ft2, ft3.to_list())], np.int32)

# 1. today's float64 walk with frame-tier precomputed bool cast to float64
pre = ft3.str.contains(pat).to_numpy().astype(np.float64)
feats = np.column_stack([ft1, ft2, pre])
prog = K.owner_shape((IS_TRUE, F64, 2, 0, 0, 4, 3, 0))
out = np.zeros(n, np.int32)
K.walk_f64(feats, np.array([5.0, 10.0]), *prog.arrays_f64(), out)
assert (out == oracle).all(), "f64 walk mismatch"

# 2. typed walk, lazy STR node calling C
h = SM.compile_pattern(pat)
print("jit:", SM.is_jit(h), "match_py('doge')=", SM.match_py(h, "doge"), "match_py('cat')=", SM.match_py(h, "cat"))
offs, vals = K.string_buffers(ft3)
inp = K.empty_typed_inputs(n)
inp.update(f64s=np.column_stack([ft1, ft2]), str_tab=K.string_table([(offs, vals)]),
           thr_f64=np.array([5.0, 10.0]), handles=np.array([h], np.int64), fn_table=np.array([SM.MATCH_ADDR], np.uint64))
prog = K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0)).validate(2, 0, 0, 0, 1, 2, 0, 1, 0)
out2 = np.zeros(n, np.int32)
K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out2)
assert (out2 == oracle).all(), "typed lazy walk mismatch"

# 3. typed walk, per-category mask
cats = ft3.cast(pl.Categorical)
codes = cats.to_physical().cast(pl.Int32).to_numpy()
cat_strs = cats.cat.get_categories()
mask = cat_strs.str.contains(pat).to_numpy().astype(np.uint8)[None, :]
inp3 = dict(inp); inp3.update(i32s=codes.reshape(-1, 1).copy(), masks=mask)
prog3 = K.owner_shape((STR_MASK, I32, 0, 0, 0, 4, 3, 0)).validate(2, 0, 0, 1, 1, 2, 0, 1, 1)
out3 = np.zeros(n, np.int32)
K.walk_typed(*[inp3[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog3.arrays(), out3)
assert (out3 == oracle).all(), "typed mask walk mismatch"
print("all three agree with oracle over", n, "rows; positives:", int(oracle.sum()))
