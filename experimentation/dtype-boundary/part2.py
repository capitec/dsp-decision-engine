"""
EXPERIMENT A, part 2 - the candidate admissible-dtype paths.

Part 1 (dtype_boundary.py) showed which dtypes reach numba at all. This tests the
four workarounds that part 1's tables imply, before recommending any of them:

  P1  string  -> Categorical/Enum physical codes (uint32/uint8).  Are the codes
      STABLE across frames?  A decision engine compares a code against a constant.
  P2  Decimal -> raw Int128 buffer.  Is there any numpy/numba path at all?
  P3  sliced series -> does zero-copy to_numpy give the correct window?
  P4  null slot -> harder attempts to observe non-zero garbage.
  P5  Int64 + null -> to_numpy lossiness, exactly.

Run:  <repo>/.venv/bin/python experimentation/dtype-boundary/part2.py
"""
import ctypes
import numpy as np
import polars as pl
from numba import njit


def line(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def sx(e, n=170):
    m = " ".join(str(e).split())
    return f"{type(e).__name__}: {m[:n]}"


# ---------------------------------------------------------------- P1
line("P1  string -> categorical/enum physical codes: are codes stable across frames?")

f1 = pl.DataFrame({"sector": ["PRIVATE", "PUBLIC", "SME"]})
f2 = pl.DataFrame({"sector": ["SME", "PRIVATE", "PUBLIC"]})   # different first-seen order
f3 = pl.DataFrame({"sector": ["PUBLIC", "PUBLIC", "PUBLIC"]})  # only one level present

for nm, f in (("frame1 PRIVATE,PUBLIC,SME", f1), ("frame2 SME,PRIVATE,PUBLIC", f2),
              ("frame3 PUBLIC only", f3)):
    c = f["sector"].cast(pl.Categorical)
    codes = c._get_buffers()["values"].to_numpy()
    print(f"  Categorical {nm:<28} values={list(f['sector'])} codes={list(codes)} dtype={codes.dtype}")

E = pl.Enum(["PRIVATE", "PUBLIC", "SME"])
for nm, f in (("frame1", f1), ("frame2", f2), ("frame3", f3)):
    c = f["sector"].cast(E)
    codes = c._get_buffers()["values"].to_numpy()
    print(f"  Enum        {nm:<28} values={list(f['sector'])} codes={list(codes)} dtype={codes.dtype}")

# unseen level
try:
    bad = pl.Series("s", ["PRIVATE", "GOVERNMENT"]).cast(E)
    print(f"  Enum unseen level -> {list(bad)}")
except BaseException as e:
    print(f"  Enum unseen level -> {sx(e)}")
try:
    badc = pl.Series("s", ["PRIVATE", "GOVERNMENT"]).cast(pl.Categorical)
    print(f"  Categorical unseen level -> codes={list(badc._get_buffers()['values'].to_numpy())} (accepted silently)")
except BaseException as e:
    print(f"  Categorical unseen level -> {sx(e)}")

# does an njit kernel actually compare codes?
@njit(cache=False)
def is_private(codes, target):
    n = 0
    for i in range(codes.shape[0]):
        if codes[i] == target:
            n += 1
    return n

for nm, arr in (("uint32 (Categorical)", f1["sector"].cast(pl.Categorical)._get_buffers()["values"].to_numpy()),
                ("uint8  (Enum)", f1["sector"].cast(E)._get_buffers()["values"].to_numpy())):
    try:
        print(f"  njit compare on {nm}: count(code==0) = {is_private(arr, arr.dtype.type(0))}")
    except BaseException as e:
        print(f"  njit compare on {nm}: {sx(e)}")

# ---------------------------------------------------------------- P2
line("P2  Decimal -> is there any path to a numba-consumable representation?")

d = pl.Series("m", ["1234.56", "0.01", "999999.99"], dtype=pl.Decimal(18, 2))
print(f"  dtype={d.dtype}  scale={d.dtype.scale}  precision={d.dtype.precision}")
vb = d._get_buffers()["values"]
print(f"  values buffer series dtype = {vb.dtype}")
for label, fn in (("values_buf.to_numpy()", lambda: vb.to_numpy()),
                  ("series.to_numpy()", lambda: d.to_numpy()),
                  ("cast(Float64).to_numpy()", lambda: d.cast(pl.Float64).to_numpy()),
                  ("cast(Int64) direct", lambda: d.cast(pl.Int64).to_numpy())):
    try:
        r = fn()
        print(f"  {label:<26} -> dtype={r.dtype} first3={list(r[:3])}")
    except BaseException as e:
        print(f"  {label:<26} -> {sx(e, 110)}")
# scaled-integer path: multiply by 10**scale in polars, then Int64
try:
    scaled = (d.cast(pl.Float64) * 100).round(0).cast(pl.Int64)
    print(f"  scaled-int via Float64*100 -> {list(scaled.to_numpy())} (exact for these values: "
          f"{list(scaled.to_numpy()) == [123456, 1, 99999999]})")
except BaseException as e:
    print(f"  scaled-int via Float64 -> {sx(e)}")
# raw pointer read of the Int128 buffer
try:
    ptr, off, ln = vb._get_buffer_info()
    raw = (ctypes.c_ubyte * (16 * ln)).from_address(ptr)
    b = bytes(raw)
    vals = [int.from_bytes(b[i*16:(i+1)*16], "little", signed=True) for i in range(ln)]
    print(f"  raw Int128 ctypes read     -> {vals}  (scale {d.dtype.scale} => "
          f"{[v / 10**d.dtype.scale for v in vals]})")
    as_i64 = np.frombuffer(b, dtype=np.int64)[::2].copy()
    print(f"  int128 low-64 lanes as i64 -> {list(as_i64)} (valid only while |value| < 2**63)")
except BaseException as e:
    print(f"  raw Int128 read -> {sx(e)}")

# float64 exactness of money at realistic magnitudes
cents = [123456, 99999999, 1, 250000000000, 9007199254740993]
print("  float64 round-trip of cent-integers:")
for c in cents:
    f = float(c) / 100.0
    back = round(f * 100)
    print(f"    {c:>20} cents -> float {f!r} -> back {back:>20}  {'OK' if back == c else 'LOSSY'}")

# ---------------------------------------------------------------- P3
line("P3  sliced series -> does zero-copy to_numpy give the correct window?")

base = pl.Series("x", [float(i) for i in range(20)])
for desc, s in (("base", base), ("slice(9,3)", base.slice(9, 3)),
                ("slice(9,3).rechunk()", base.slice(9, 3).rechunk()),
                ("head(5)", base.head(5)), ("tail(3)", base.tail(3))):
    try:
        z = s.to_numpy(allow_copy=False)
        ptr, off, ln = s._get_buffer_info()
        print(f"  {desc:<22} to_numpy(no-copy)={list(z)}  buffer_info=(ptr+{ptr - base._get_buffer_info()[0]}, off={off}, len={ln})"
              f"  correct={list(z) == list(s)}")
    except BaseException as e:
        print(f"  {desc:<22} {sx(e)}")

# two-chunk case
try:
    ch = pl.concat([pl.Series("x", [1.0, 2.0, 3.0]), pl.Series("x", [7.0, 8.0])], rechunk=False)
    print(f"  2-chunk series n_chunks={ch.n_chunks()}")
    try:
        z = ch.to_numpy(allow_copy=False)
        print(f"    to_numpy(allow_copy=False) -> {list(z)} (SUCCEEDED across chunks)")
    except BaseException as e:
        print(f"    to_numpy(allow_copy=False) -> {sx(e, 110)}")
    print(f"    _get_buffers() on 2-chunk -> ", end="")
    try:
        b = ch._get_buffers()
        print(f"values len={len(b['values'])} content={list(b['values'])} (series len={len(ch)}, "
              f"values={list(ch)}) -> _get_buffers SILENTLY COPIES/rechunks; only allow_copy=False "
              f"is an honest zero-copy gate")
    except BaseException as e:
        print(sx(e, 110))
except BaseException as e:
    print(f"  chunk test {sx(e)}")

# ---------------------------------------------------------------- P4
line("P4  null slot: harder attempts to observe non-zero garbage")

cases = {}
L = pl.DataFrame({"k": list(range(10))})
R = pl.DataFrame({"k": [0, 2, 4, 6, 8], "v": [11.0, 22.0, 33.0, 44.0, 55.0]})
cases["left join (10x5)"] = L.join(R, on="k", how="left")["v"]
big_l = pl.DataFrame({"k": list(range(1000))})
big_r = pl.DataFrame({"k": list(range(0, 1000, 3)), "v": [float(i) for i in range(0, 1000, 3)]})
cases["left join (1000)"] = big_l.join(big_r, on="k", how="left")["v"]
cases["shift(2)"] = pl.Series("s", [float(i) for i in range(10)]).shift(2)
cases["gather with null idx"] = pl.Series("s", [float(i) for i in range(10)]).gather([1, 2, 3]).append(pl.Series("s", [None], dtype=pl.Float64))
cases["set_at 3 -> null"] = pl.Series("s", [7.0] * 10).scatter([3, 5], None)
cases["filter+reindex"] = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}).with_columns(
    pl.when(pl.col("a") > 2).then(pl.col("a")).otherwise(None).alias("b"))["b"]
cases["when/otherwise null"] = pl.Series("s", [5.0, 6.0, 7.0]).set(pl.Series([False, True, False]), None)

for name, s in cases.items():
    try:
        b = s._get_buffers()
        raw = b["values"].to_numpy()
        py = list(s)
        nulls = [i for i, x in enumerate(py) if x is None]
        under = [float(raw[i]) for i in nulls]
        nz = [u for u in under if u != 0.0]
        print(f"  {name:<24} nulls={len(nulls):<4} under-null slots (first 6)={under[:6]}  NON-ZERO: {len(nz)}")
    except BaseException as e:
        print(f"  {name:<24} {sx(e, 110)}")

# ---------------------------------------------------------------- P5
line("P5  Int64 + null -> to_numpy lossiness, exactly")
probes = [2**53, 2**53 + 1, 2**60 + 1, 2**62 + 12345, -(2**60) - 7]
s = pl.Series("i", probes + [None], dtype=pl.Int64)
a = s.to_numpy()
print(f"  clean Int64 to_numpy dtype = {pl.Series('i', probes, dtype=pl.Int64).to_numpy().dtype}")
print(f"  +null Int64 to_numpy dtype = {a.dtype}")
for i, p in enumerate(probes):
    got = a[i]
    exact = int(got) == p if np.isfinite(got) else False
    print(f"    {p:>22} -> {got!r:>26}  {'EXACT' if exact else 'LOSSY (delta %d)' % (int(got) - p)}")
vb = s._get_buffers()["values"].to_numpy()
print(f"  values-buffer path dtype = {vb.dtype}, exact = {list(vb[:len(probes)]) == probes}")
