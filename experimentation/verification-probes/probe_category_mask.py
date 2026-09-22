"""Independent check of the headline recommendation: the per-category mask.

Claim under test (Rust strand): on a low-cardinality string column, matching a
pattern by running it over the DISTINCT values once and then doing one array
lookup per row costs 0.8-1.8 ns/row, beating every in-kernel strategy.

Measured here end to end, and honestly: the lookup, the mask build, and the
dictionary encode if the column is not already encoded.
"""
import json, re, time
import numpy as np, polars as pl
from numba import njit

N = 1_000_000
CATS = ["retail","mining","dogma","dogfood","finance","agri","dogwalker",
        "logistics","telecom","utilities","dogsbody","education"]
rng = np.random.default_rng(11)
codes_np = rng.integers(0, len(CATS), N).astype(np.int32)
col = pl.Series("employer", [CATS[c] for c in codes_np])

PATTERN = r"^dog"
oracle = np.array([bool(re.search(PATTERN, CATS[c])) for c in codes_np])

def t(fn, reps=5):
    fn()
    return min((lambda: (lambda s: (fn(), time.perf_counter()-s)[1])(time.perf_counter()))() for _ in range(reps))

# --- 1. the mask build: regex over the DISTINCT values only
def build_mask():
    distinct = col.unique().to_list()
    return {v: bool(re.search(PATTERN, v)) for v in distinct}
ms_mask = t(build_mask) * 1e3

# --- 2. the dictionary encode, if the column is not already encoded
def encode():
    return col.cast(pl.Categorical).to_physical().to_numpy()
ms_encode = t(encode) * 1e3
phys = encode()
mask = np.zeros(phys.max() + 1, dtype=np.uint8)
cat_order = col.cast(pl.Categorical).cat.get_categories().to_list()
for i, v in enumerate(cat_order):
    mask[i] = bool(re.search(PATTERN, v))

# --- 3. the per-row lookup inside a kernel
@njit(cache=False)
def walk_mask(codes, mask, out):
    for i in range(out.shape[0]):
        out[i] = mask[codes[i]]

out = np.zeros(N, dtype=np.uint8)
walk_mask(phys.astype(np.int32), mask, out)
assert (out.astype(bool) == oracle).all(), "mask answer wrong"
ns_lookup = t(lambda: walk_mask(phys.astype(np.int32), mask, out)) / N * 1e9

# --- 4. the incumbent: polars str.contains over every row
def frame():
    return col.str.contains(PATTERN).to_numpy()
ms_frame = t(frame) * 1e3
assert (frame() == oracle).all()

print(f"rows                                    : {N:,}   distinct: {col.n_unique()}")
print(f"per-row mask lookup in kernel           : {ns_lookup:6.2f} ns/row")
print(f"mask build (regex over distinct values) : {ms_mask:6.2f} ms/batch  = {ms_mask*1e6/N:.2f} ns/row")
print(f"dictionary encode, if not already done  : {ms_encode:6.2f} ms/batch  = {ms_encode*1e6/N:.2f} ns/row")
print(f"polars str.contains over every row      : {ms_frame:6.2f} ms/batch  = {ms_frame*1e6/N:.2f} ns/row")
print()
print(f"TOTAL, codes already paid for           : {ns_lookup + ms_mask*1e6/N:6.2f} ns/row")
print(f"TOTAL, including the encode             : {ns_lookup + (ms_mask+ms_encode)*1e6/N:6.2f} ns/row")
print(f"incumbent (frame tier)                  : {ms_frame*1e6/N:6.2f} ns/row")

# --- 5. the case decider2 is actually in: the column ALREADY arrives encoded
print()
print("--- column already Categorical (decider2 dictionary-encodes at the boundary) ---")
cat_col = col.cast(pl.Categorical)
def mask_from_categorical():
    cats = cat_col.cat.get_categories().to_list()
    m = np.zeros(len(cats), dtype=np.uint8)
    for i, v in enumerate(cats):
        m[i] = bool(re.search(PATTERN, v))
    return m
us_mask2 = t(mask_from_categorical) * 1e6
def codes_from_categorical():
    return cat_col.to_physical().to_numpy()
us_codes = t(codes_from_categorical) * 1e6
print(f"mask build from existing categories     : {us_mask2:8.1f} us/batch = {us_mask2*1e3/N:.4f} ns/row")
print(f"get the int32 codes out                 : {us_codes:8.1f} us/batch = {us_codes*1e3/N:.4f} ns/row")
print(f"TOTAL with codes already in hand        : {ns_lookup + (us_mask2+us_codes)*1e3/N:6.2f} ns/row")
print(f"  vs incumbent frame tier               : {ms_frame*1e6/N:6.2f} ns/row")
