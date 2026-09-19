# Experiment C — numba's on-disk cache, and what actually invalidates it

## What this measures

`decider2` plans to compile at Docker-image build time and have a runtime load trigger
**zero** compilations (doc 02 §3.4, doc 05 §8). That only works if a cache written at build
is still *hit* at runtime. This harness measures, on real numba, exactly which conditions
produce a hit and which produce a miss.

It tests four claims:

| claim | source |
|---|---|
| "Numba cannot cache code without a source file (`RuntimeError: no locator available for file '<string>'`)" | doc 05 §4.1 |
| "Numba's cache is keyed on source. Generated code must be **byte-identical** between build and runtime or every entry misses" | doc 05 §4.2 |
| "numba's file-backed cache stamps entries with `(st_mtime, st_size)` of the generated `.py` and indexes by path, so regenerating a byte-identical file at a different path or time misses 100%" | REVIEW.md §7 |
| "Numba's cache keys include CPU features … `NUMBA_CPU_NAME=generic` at build *and* run makes entries portable" | doc 05 §8 |

## How to run

```sh
# the full matrix (~2.5 min) plus the cold-vs-warm size sweep (~1.5 min)
.venv/bin/python experimentation/numba_cache_survival/run_experiment.py --scaling

# matrix only
.venv/bin/python experimentation/numba_cache_survival/run_experiment.py

# just the forged-stamp / changed-constant safety check (~15 s)
.venv/bin/python experimentation/numba_cache_survival/run_experiment.py --only stale

# a bigger driver (compile cost grows superlinearly with argument count)
.venv/bin/python experimentation/numba_cache_survival/run_experiment.py --n-args 400
```

Results land in `results.json` next to the script. The generated drivers and their caches are
left in a temp workdir, printed at the end, so you can inspect the `.nbi`/`.nbc` files yourself.

## Files

- `gen_driver.py` — deterministic codegen for the driver under test: one `@njit(cache=True)`
  function with `n_args` scalar arguments and one real branch per argument. Stands in for
  decider2's generated fused driver. `gen_source(n, comment)` is byte-stable.
- `child.py` — runs **one** scenario in a **fresh process** and prints JSON. This is the whole
  point: numba's in-memory dispatcher cache hides every on-disk effect within one process.
  It reports `dispatcher.stats.cache_hits/misses` (numba's own counters, not a timing guess),
  the resolved cache path, the index path, the locator class, and the `(st_mtime, st_size)`
  source stamp.
- `run_experiment.py` — the orchestrator and the invalidation matrix.

## Why the measurement is trustworthy

- Hit/miss is read from numba's own `Dispatcher.stats` counters, and cross-checked against the
  first-call wall time (a miss costs seconds, a hit ~0.18 s).
- Every scenario is a separate `subprocess`, so nothing is warm in memory.
- Timed repeats: the warm case is the median of 3 fresh processes. The already-compiled call is
  warmed once then timed 5 times.
- Byte-identity is asserted with SHA-256 in the harness itself, not assumed.
- The stale-code check verifies that the edit it makes actually changes the computed result
  (by wiping the cache and recompiling) before it claims staleness.

## Scenario key

| id | setup |
|---|---|
| A1/A2 | fresh build, then the same untouched file re-run |
| B | byte-identical file rewritten at the same path (mtime moves) |
| C | byte-identical rewrite, then `os.utime` restores the old mtime |
| D | `os.utime` only; bytes never touched |
| E | byte-identical file in a different directory, no cache copied |
| F | the whole build directory copied with mtimes preserved (`cp -p`) |
| G | same content and cache, file renamed |
| H | a comment changed, so the file size changes |
| I | a comment **line** prepended, so the function's first line number shifts |
| L | a comment changed to a **same-length** string, mtime restored — bytes differ, stamp does not |
| M | a **constant** changed to a same-length one, mtime restored — does numba serve the old machine code? |
| J1–J5 | `NUMBA_CPU_NAME=generic` against a native-built cache, and back again |
| K1–K5 | driver imported *without* registering it in `sys.modules` |

## Measured (2026-09-18, numba 0.67.0 / llvmlite 0.49.0 / CPython 3.14.5, x86_64 raptorlake)

Full output in `results.json`. Re-run to get your own numbers; do not cite these for another
machine, another numba, or another driver shape.

### Cold compile vs cached load (driver = N args, one branch each)

| n_args | cold (miss) | warm, median of 3 fresh processes | speedup | `.nbc` size |
|---|---|---|---|---|
| 200 | 4.93 s | 0.177 s | 27.8x | 526 KiB |
| 400 | 14.75 s | 0.179 s | 82.5x | 1060 KiB |
| 800 | 46.99 s | 0.197 s | 238.7x | 2130 KiB |

Warm cost is flat at ~0.18 s regardless of driver size; `import numba` itself is ~0.4 s.

### Invalidation matrix (n_args = 200)

| scenario | bytes identical? | stamp identical? | result |
|---|---|---|---|
| A2 untouched file | yes | yes | **HIT** 0.183 s |
| B byte-identical rewrite (new mtime) | yes | no | **MISS** 4.67 s |
| C identical rewrite + `os.utime` restore | yes | yes | **HIT** 0.177 s |
| D `touch` only | yes | no | **MISS** 4.80 s |
| E identical bytes, different dir, no cache | yes | n/a | **MISS** 4.61 s |
| F whole dir copied, mtimes preserved | yes | yes | **HIT** 0.171 s |
| G same content + cache, file renamed | yes | yes | **MISS** 4.80 s |
| H comment changed (size changes) | no | no | **MISS** 4.64 s |
| I comment **line** prepended | no | no | **MISS** 4.72 s |
| L comment changed, same length, mtime restored | **no** | yes | **HIT** 0.179 s |
| M constant changed, same length, mtime restored | **no** | yes | **HIT — served stale machine code** |

M is the dangerous one: served 49.39775000000002 (the pre-edit result) where a forced
recompile of the same file gives 49.29765000000001.

### The real rule

A hit requires **all** of:
1. same absolute directory and same file basename (they pick the `__pycache__` dir and the
   `.nbi` name), and the same `def` line number (it is the filename disambiguator);
2. `(st_mtime, st_size)` of the `.py` exactly equal to the stamp written at build;
3. same argument signature;
4. same `magic_tuple` = (LLVM triple, CPU name, CPU feature string);
5. same sha256 of the function's `co_code` **and** of its pickled closure;
6. the module is in `sys.modules` under the name it was pickled with, at load time.

Source *bytes* appear nowhere. They matter only through `st_size`, the line number, and
`co_code` — and `co_code` does not cover `co_consts`, which is why M hits with the wrong maths.
