"""Independent check that polars' Arrow C stream export does not copy a String
column. Two witnesses that need no knowledge of the layout:

  RSS   -- a copy of N rows of 13-22 byte strings must show up as tens of MB.
  TIME  -- a copy is O(rows); a handshake is O(1).

Compared against _get_buffers(), which is known to copy, as a positive control.
"""
import gc, os, resource, time
import polars as pl

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def make(n):
    return pl.DataFrame({"s": [f"merchant-{i:09d}" for i in range(n)]})  # 19 bytes each

def timeit(fn, reps=5):
    fn()
    return min((lambda t0: (fn(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(reps))

print(f"{'rows':>10s} {'arrow_c_stream':>16s} {'_get_buffers':>16s}   {'ns/row (stream)':>16s} {'ns/row (buffers)':>17s}")
for n in (100_000, 1_000_000, 4_000_000):
    df = make(n); s = df["s"]
    t_stream = timeit(lambda: s.__arrow_c_stream__())
    t_buf = timeit(lambda: s._get_buffers(), reps=3)
    print(f"{n:10,d} {t_stream*1e6:14.1f}us {t_buf*1e6:14.1f}us   {t_stream/n*1e9:14.3f} {t_buf/n*1e9:16.1f}")
    del df, s; gc.collect()

print()
n = 4_000_000
df = make(n); s = df["s"]
gc.collect(); base = rss_mb()
caps = [s.__arrow_c_stream__() for _ in range(5)]
gc.collect(); after_stream = rss_mb()
bufs = [s._get_buffers() for _ in range(5)]
gc.collect(); after_buffers = rss_mb()
print(f"{n:,} rows x 19 bytes = ~{n*19/1e6:.0f} MB of string data")
print(f"  peak RSS baseline                 : {base:8.1f} MB")
print(f"  after 5x __arrow_c_stream__()     : {after_stream:8.1f} MB   (delta {after_stream-base:+.1f})")
print(f"  after 5x _get_buffers()           : {after_buffers:8.1f} MB   (delta {after_buffers-after_stream:+.1f})")
print()
print("A copy of the column would add tens of MB per call; a handshake adds ~0.")
