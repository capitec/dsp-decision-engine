# Numba compile cache

**Decision:** Anything compiled with `cache=True` lives in a real `.py` file,
never `exec`, and is imported by module name. Generated code is byte-identical
across runs, and its file and module names come from a hash of its content. Record
the CPU target with the cache and check it at startup.

**Why:**
- `exec`'d code can't be cached at all. It fails when decorated:
  `RuntimeError: cannot cache function 'f': no locator available for file '<string>'`.
- A warm cache turns compile time into a build-time cost. Cold vs warm in a fresh
  process: 4.93 s vs 0.177 s at 200 args, 14.75 s vs 0.179 s at 400, and 46.99 s vs
  0.197 s at 800. Warm cost stays flat at about 0.18 s whatever the driver size.
- A cache entry is used only if all seven conditions hold:
  1. same absolute directory;
  2. same basename;
  3. same `def` line number;
  4. exact `(st_mtime, st_size)`;
  5. same argument signature;
  6. same `magic_tuple` (LLVM triple, CPU name, CPU features);
  7. same `sys.modules` name.
- If the source is regenerated on every start, the mtime changes and every entry
  misses.
- **Stale code is served silently.** numba's index hashes `co_code`, which
  excludes `co_consts`. An edit that replaces a constant used only once keeps the
  file size, and with a restored mtime numba reports a HIT and returns the
  pre-edit answer (49.39775 instead of 49.29765). CPython's `.pyc` can also
  serve stale code, independently. Content-addressed file names close both
  layers.
- The CPU name and features are in the cache key. A cache built on an AVX-512 CI
  runner misses on a smaller deployment host. The miss is silent, and the service
  recompiles at startup.

**What we tried:**
- Normalising mtimes for reproducible builds. It turns the stale-constant problem
  into a silent wrong-decision bug, and `build --verify` still reports success.
- Loading generated files with `importlib.util.spec_from_file_location`. The cache
  never survived a fresh process; importing by module name took it from 12.1 s
  cold to about 0.2 s.
- A mismatched `sys.modules` name raises `ModuleNotFoundError('<dynamic>')` from
  inside numba's `pickle.loads`.
- `NUMBA_CPU_NAME=generic` makes entries portable, but it gives up the
  instruction selection that small kernels depend on.
- decider2 later stopped writing driver source. Each step is `njit(cache=True)` in
  its own file. The fused glue closes over dispatchers, so it is built per process
  and not disk-cached: a closure over a dispatcher never hits and grows the index.
  `precompile()`/`warm()` keeps that compile off the request path.

**Source:** `decider2/docs/EXPERIMENTS.md` (§C, §K, §M, §J2);
`decider2/docs/05-boundary-and-compilation.md` (§4.1, §4.2, §8);
`experimentation/numba_cache_survival/`; `experimentation/kl-contradiction-resolved/`;
`experimentation/subprocess-handover/`; `decider2/src/decider2/compile/kernel.py`
