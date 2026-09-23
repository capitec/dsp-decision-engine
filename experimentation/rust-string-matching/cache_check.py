"""The decisive non-performance question: does a kernel that calls Rust
through a pointer-table ARGUMENT still numba-disk-cache? Methodology as
EXPERIMENTS.md §V/§W: a persistent dir (`__pycache__/` next to
kernels.py, numba's default for an importable module), `NUMBA_DEBUG_CACHE=1`,
one process "cold", a SEPARATE process "warm". Warm must show
`[cache] ... loaded` and NO `saved`.

    NUMBA_DEBUG_CACHE=1 python cache_check.py cold|warm
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    mode = sys.argv[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        from binding import StringBuffers, compile_pattern, make_ptr_table
        from data import make_data, thresholds
        import kernels
        df, _ = make_data(10_000, "low", seed=1)
        sb = StringBuffers(df["ft3"])
        ft1, ft2 = df["ft1"].to_numpy(), df["ft2"].to_numpy()
        out = np.empty(len(ft1), dtype=np.bool_)
        PT = make_ptr_table(); pid = np.int32(compile_pattern("^dog"))
        c1, c2 = thresholds(0.5)
        hits = kernels.lazy_rust(ft1, ft2, sb.offsets, sb.values, PT, pid, c1, c2, out)
        elapsed = time.perf_counter() - t0
    cache_warnings = [str(w.message) for w in caught if "cach" in str(w.message).lower()]
    nbi = sorted(p.name for p in (HERE / "__pycache__").glob("kernels.lazy_rust*.nbi"))
    print(f"[{mode}] import+build+first call: {elapsed*1000:.1f} ms; hits={hits}; "
          f"cache warnings={cache_warnings or 'none'}; lazy_rust index files={nbi}")
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps({"experiment": "rust-string-matching", "item": "cache_check", "mode": mode,
                            "elapsed_ms": elapsed * 1000, "cache_warnings": cache_warnings, "nbi": nbi}) + "\n")


if __name__ == "__main__":
    main()
