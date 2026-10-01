# Thread-safety audit of `decider/` (2026-09-30)

An audit of global state and state shared between multiple executions of the
flow. The axis is the serving path: `RequestHandler` holds one `_active` `Live`
with a single `Executable`, and every request calls `live.executable.score()` /
`run()` on it (`serving/handler.py`). The codebase itself expects concurrent
calls on one pipeline — `boundary/extract.py` pools `FrameView`s per thread with
the comment "serving runs concurrent calls on one pipeline" — so anything
mutable on the `Executable`/runner is shared state between executions. Findings
are ranked by severity; nothing was changed.

## 1. HIGH — `SteppedRunner._bundle`: span-buffer lifetime race

`engine/run/runners/stepped.py:_bundle` (`_converted`/`_alive`, lines 60-62 and
177-185) does an unsynchronized read-then-write on every run of a node whose
`str` params enter the kernel as UTF-8 spans:

```python
self._alive[key] = utf8          # keeps the bytes the spans point into alive
converted = self._converted[key] = bundle._replace(**pointers)
```

The `utf8` buffers are kept alive only by `self._alive[key]`; the returned
bundle carries raw `(addr, len)` pointers. For the common serving case — one
shared, immutable params document — every concurrent request collides on the
same `key`. If thread B overwrites `self._alive[key]` between A's store and A's
kernel reading the spans, A's buffer can be GC'd and A's kernel reads freed
memory.

- Impact: use-after-free / reads of garbage, not merely a lost update.
- Fix direction: a per-thread or per-call converted-bundle cache, mirroring the
  `_local` pool already in `boundary/extract.py`.

## 2. MEDIUM-HIGH — `Executable` / `ParamsCache` are shared mutable state mutated on the hot path

- `engine/run/engine.py:103-108, 150, 164, 170` — `self.cache`, `self._checked`,
  `self._warned`, `self.report` are all mutated inside
  `prepare()`/`_params()`/`_check_shadowing()`, unlocked. The code admits it at
  lines 105-107 ("concurrent callers race on it").
- `engine/params/validate.py:135-179` — `ParamsCache._results` (read-then-write
  in `validate`, line 178) and the single-slot `_last` (`key`, lines 164-168).
  `key()` is correct under races (it re-checks `last[0] is doc`), but `_results`
  read-modify-write races produce duplicate validation / lost updates of the
  report, and the `_last` slot thrashes when callers alternate documents.
- `steps/each.py:90, 103, 127` — `_EachRunner` reuses one `ParamsCache` across
  all parent rows and, via the shared executable, across concurrent runs.

- Impact: `exe.report` is last-writer-wins (wrong report under concurrency);
  duplicate `check_namespaces`/validation; possible duplicate
  `FallbackWarning`s. Mostly wasteful rather than corrupt, but genuine
  unsynchronized shared state.
- Fix direction: move `self.report`/`_checked` mutation onto the per-call
  `RunParams` rather than the shared `Executable`.

## 3. MEDIUM — compile-path module-global caches: check-then-act with no lock

All follow `x = D.get(k); if x is None: D[k] = build(...)`, hit on first
compilation (and re-compilation):

- `engine/compile/njit.py:43-44` `_DISPATCHERS`, `_REASONS`; `83-89`
  (`_DISPATCHERS[key] = njit(fn)`), `146-152` (`dispatcher.compile` +
  `_REASONS[...] = ...`), and `89` (`dispatcher._cache = _SaltedCache(fn)` on a
  shared dispatcher object).
- `engine/compile/kernel.py:94, 117-125` `_KERNELS` / `fused_kernel`
  (`njit(nogil=True)`).
- `engine/compile/structs.py:23, 44-51` `_DTYPES` — also reachable on the read
  path via `build_struct`.
- `engine/compile/rows.py:38, 41-49` `_LAYOUTS` — hot path via
  `build_rows`/`build_ragged` for `Columnar[Item]`.

- Impact: under the GIL, dict `get`/`setitem` are individually atomic and numba
  serializes actual compilation with its own lock, so these races yield
  duplicate work and last-writer-wins — benign in content but unsynchronized,
  and `dispatcher._cache = ...` on a shared dispatcher is an unlocked attribute
  swap.

## 4. MEDIUM — runner lazy-compile and packed-dict mutation

- `engine/run/runners/stepped.py:66-68` `if plan is not self._plan:
  self._compile(...)` is check-then-act; two threads on a first call both
  compile and stomp `self.units/_reads/_converted/_alive/_plan`.
- `engine/run/runners/fused.py:35, 79` — `self.packed.pop(path)` during
  `_packed`, and `_compile` replaces `self.packed` (`39`).
- `steps/each.py:141-149` `_BatchRunner` lazy
  `if self.exe is None: self.exe = Engine().bind(...)` — classic check-then-act.

## 5. LOW — registry maps mutated at runtime

`registry/core.py:55-57, 75-86, 108-118, 133-139`. `_registry`/`_lazy` are
ClassVars populated at class-definition time, but also mutated at runtime by
`resolve()` (importing a lazy module triggers registration) and `lazy()`.

- Impact: idempotent per `(alias, path)` and `importlib.import_module` is
  internally locked, so worst case is a redundant import/registration if configs
  are loaded concurrently (startup or hot-reload). Benign in practice.

## 6. LOW / benign — the rest

- `types.py:12, 178-182` `_RAW_STR_CODES` — double-checked lock; the fast-path
  `_RAW_STR_CODES.get(value)` read is outside `_NEW_CODE`, but dict `get` is
  GIL-atomic and the re-check is inside the lock. Correct.
- `boundary/extract.py:81, 85-96` — `lru_cache` (thread-safe) +
  `threading.local()` view pool. The correct pattern; the only place per-thread
  state is handled right. (`_schema_plan` is `maxsize=None`, unbounded but safe.)
- `serving/handler.py:33` — `Live` has a mutable default `dates: dict = {}` in a
  NamedTuple. Never used in practice (`stage()` always passes `_dates(exe)`) and
  never mutated, but it is a shared mutable default.
- `engine/run/runners/interpreted.py:63` — `skip: dict = {}` is a class-level
  mutable default, but it is only ever *replaced* by instance assignment in
  debug sessions (`debug/edit.py:206-210`), never mutated in place, so the
  shared dict is a read-only sentinel.
- `engine/run/engine.py:31` `_NO_FRAME = pl.DataFrame()` — shared frame, but the
  `_record_path` branch guarantees no frame step mutates it.

## Bottom line

Finding 1 (`_bundle` span lifetime) is the dangerous one. Finding 2 is the most
likely to bite observability (`exe.report`) and warning correctness. Findings 3
and 4 are the standard "lazy compile + shared dispatcher caches" races, mostly
wasteful under the GIL. Everything else is benign.
