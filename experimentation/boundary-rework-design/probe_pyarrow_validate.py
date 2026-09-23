"""Appendix probe: does Arrow C++ (pyarrow) validate string-view ELEMENTS
where nanoarrow does not?  Each case runs in its own subprocess: a polars
String column is exported zero-copy to pyarrow (`Series.to_arrow()` goes
through the C Data Interface), the view of row 1 is corrupted IN PLACE in
polars' own buffer, then `validate()` and `validate(full=True)` are called
and the exit code / exception is recorded.  pyarrow was pip-installed into
the venv for this probe only (it is not a project dependency)."""
from __future__ import annotations
import ctypes, json, subprocess, sys

STRINGS = ["dog", "a much longer merchant descriptor dog", "cat", None, "x" * 40 + "dog", "hotdog"]
CASES = ["clean", "bad_buffer_index", "bad_offset", "bad_length", "neg_length", "sliced_clean"]

CHILD = r'''
import ctypes, sys, json
import polars as pl, pyarrow as pa
case = sys.argv[1]
STRINGS = %r
def poke_u32(addr, v): ctypes.c_uint32.from_address(addr).value = v
if case == "sliced_clean":
    s = pl.Series("s", ["p0","p1","p2"] + STRINGS + ["p3"]).slice(3, len(STRINGS))
elif case == "null_validity_nonzero_null_count":
    s = pl.Series("s", [x or "none" for x in STRINGS])
else:
    s = pl.Series("s", STRINGS)
arr = pa.chunked_array(s)  # via __arrow_c_stream__: zero-copy, keeps polars' string_view
if isinstance(arr, pa.ChunkedArray):
    assert arr.num_chunks == 1
    arr = arr.chunk(0)
assert pa.types.is_string_view(arr.type), arr.type
bufs = arr.buffers()
views = bufs[1].address + 16 * (arr.offset + 1)   # row 1: a long (out-of-line) string
if case == "bad_buffer_index": poke_u32(views + 8, 1000)
elif case == "bad_offset": poke_u32(views + 12, 2**31 - 1)
elif case == "bad_length": poke_u32(views + 0, 2**30)
elif case == "neg_length": poke_u32(views + 0, 0xFFFFFFFF)
res = {"case": case, "type": str(arr.type), "n_buffers": len(bufs)}
print("STEP start", flush=True)
for full in (False, True):
    key = "full" if full else "basic"
    try:
        arr.validate(full=full)
        res[key] = "ok"
    except Exception as e:
        res[key] = f"{type(e).__name__}: {str(e)[:160]}"
    print(f"STEP {key}={res[key]}", flush=True)
try:
    res["read_row1"] = repr(arr[1].as_py())[:40]
except Exception as e:
    res["read_row1"] = f"{type(e).__name__}: {str(e)[:100]}"
print(f"STEP read_row1={res['read_row1']}", flush=True)
print("RESULT " + json.dumps(res))
''' % (STRINGS,)

def main():
    py = sys.executable
    for case in CASES:
        p = subprocess.run([py, "-c", CHILD, case], capture_output=True, text=True, timeout=120)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            r = json.loads(line[7:])
            print(f"{case:34s} basic={r['basic']!s:60.60s} full={r['full']!s:60.60s} read_row1={r['read_row1']} ")
        else:
            steps = [l[5:] for l in p.stdout.splitlines() if l.startswith("STEP ")]
            print(f"{case:34s} CRASHED rc={p.returncode} after steps {steps}  stderr={p.stderr.strip().splitlines()[-1:] if p.stderr else ''}")

if __name__ == "__main__":
    main()
