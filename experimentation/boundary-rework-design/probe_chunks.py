"""Probe 3: chunk handling and dictionary columns at the FRAME level.
(a) a frame whose columns have different chunk layouts: what does
    df.__arrow_c_stream__() hand over -- one struct per aligned chunk, or
    does polars align/rechunk for us?
(b) a sliced frame: where does the Arrow offset live (struct or child)?
(c) Categorical / Enum children: dictionary view present, index type, dict length.
(d) how much does a multi-chunk frame cost to export at n=1 vs single chunk."""
import ctypes, time, statistics, functools, builtins
print = functools.partial(builtins.print, flush=True)
import numpy as np, polars as pl
import rowshim as rs
import arrowc_min as am
import arrowc

lib = rs.lib
P = ctypes.POINTER(arrowc.ArrowArrayStream)


def stream_chunks(df):
    cap = df.__arrow_c_stream__()
    stream = ctypes.cast(arrowc._PyCapsule_GetPointer(cap, b"arrow_array_stream"), P).contents
    schema = arrowc.ArrowSchema(); stream.get_schema(ctypes.byref(stream), ctypes.byref(schema))
    out = []
    while True:
        a = arrowc.ArrowArray()
        assert stream.get_next(ctypes.byref(stream), ctypes.byref(a)) == 0
        if not a.release: break
        kids = [(a.children[k].contents.length, a.children[k].contents.offset, a.children[k].contents.n_buffers) for k in range(a.n_children)]
        out.append(dict(length=a.length, offset=a.offset, n_children=a.n_children, children=kids))
        a.release(ctypes.byref(a))
    schema.release(ctypes.byref(schema)); stream.release(ctypes.byref(stream))
    return out

a = pl.DataFrame({"x": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]})
b = pl.DataFrame({"x": [4.0, 5.0], "s": ["d", "e"]})
df2 = pl.concat([a, b])                       # 2 chunks per column
df3 = df2.with_columns(pl.Series("y", [1, 2, 3, 4, 5]))   # y is 1 chunk, x/s are 2
print("(a) n_chunks per column:", {c: df3[c].n_chunks() for c in df3.columns}, "frame n_chunks:", df3.n_chunks("all"))
print("    exported struct chunks:", stream_chunks(df3))
print("    after export, n_chunks per column:", {c: df3[c].n_chunks() for c in df3.columns})
df2b = pl.concat([a, b]); print("    aligned 2-chunk frame: n_chunks before", df2b.n_chunks("all"), "exports:", stream_chunks(df2b), "after:", df2b.n_chunks("all"))
big = pl.concat([pl.DataFrame({"x": np.arange(3.0), "s": ["a","b","c"]})] * 4); print("    aligned 4-chunk frame exports", len(stream_chunks(big)), "struct chunk(s); after:", big.n_chunks("all"))
sl = df2[1:4]
print("(b) sliced df2[1:4]:", stream_chunks(sl), " n_chunks:", sl.n_chunks("all"))
one = pl.DataFrame({"x": np.arange(10.0), "s": [str(i) for i in range(10)]})
print("    sliced single-chunk [3:7]:", stream_chunks(one[3:7]))

# (c) dictionary children through nanoarrow
dfc = pl.DataFrame({"cat": pl.Series(["x", "y", None, "x"], dtype=pl.Categorical),
                    "en": pl.Series(["p", "q", "p", None], dtype=pl.Enum(["p", "q"])),
                    "s": ["a", None, "c", "d"]})
imp = rs.FrameImport(); imp.import_df(dfc)
for k, name in enumerate(dfc.columns):
    v = imp.child(k); d = lib.rs_dictionary(v)
    st = lib.rs_storage_type(v)
    print(f"(c) {name}: storage_type={st} ({'uint32' if st==9 else 'uint8' if st==3 else st}) dict={'yes' if d else 'no'}"
          + (f" dict_len={lib.rs_length(d)} dict_type={lib.rs_storage_type(d)} idx=[{', '.join(str(lib.rs_get_i64(v, i)) if not lib.rs_is_null(v, i) else 'null' for i in range(4))}]" if d else ""))
imp.release()

# (d) export cost at n=1 for a 17-col frame, single chunk vs two chunks (concat of two 1-row frames)
def tm(fn, reps=3000):
    t0 = time.perf_counter_ns()
    for _ in range(reps): fn()
    return (time.perf_counter_ns() - t0) / reps / 1000
cols = {f"c{i}": [1.0] for i in range(16)}; cols["s"] = ["abc"]
one = pl.DataFrame(cols); two = pl.concat([one, one]); two1 = two[0:1]
print(f"(d) n=1 polars stream walk: single-chunk 17 cols {tm(lambda: am.walk_stream(one)):.2f} us; 2-chunk frame (2 rows) {tm(lambda: am.walk_stream(two)):.2f} us; 2-chunk sliced to 1 row {tm(lambda: am.walk_stream(two1)):.2f} us")
for ncol in (1, 4, 17, 50):
    c = {f"c{i}": [1.0] for i in range(ncol)}
    print(f"    {ncol:3d} cols: stream walk {tm(lambda: am.walk_stream(pl.DataFrame(c))):.2f} us  (frame built outside? no: incl. DataFrame ctor {tm(lambda: pl.DataFrame(c), 1000):.2f} us)")
d17 = pl.DataFrame({f"c{i}": [1.0] for i in range(17)})
print(f"    17 cols, frame prebuilt: stream walk {tm(lambda: am.walk_stream(d17)):.2f} us; rechunk() {tm(lambda: d17.rechunk()):.2f} us; n_chunks('all') {tm(lambda: d17.n_chunks('all')):.2f} us")
