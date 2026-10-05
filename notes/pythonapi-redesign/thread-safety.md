# Thread safety

This file builds on `notes/racecondition_investigation_results.md` (the 2026-09-30 audit). Each finding was re-checked against the code on 2026-10-05, and three new ones were added. It turns them into a contract, a set of rules, a fix list, and tests that keep it fixed.

**The goal:** everything on the serving path is safe to call from many threads at once. Nothing else has to be. None of the fixes change an API or anything a single-threaded user sees.

## The contract

| Must be thread-safe | Why |
|---|---|
| `Executable.run` / `score` on one shared `Executable`: same or different params documents, traced or not, **including the very first call** | `RequestHandler` serves every request from one `Executable`. Kernels are `nogil=True` precisely so threads overlap (`kernel-flags-nogil-fastmath.md`). |
| `RequestHandler.process_fn` while `stage` / `activate` / `rollback` run | Hot swaps happen under load. They are already locked (`handler.py:89`); keep it that way. |
| `Engine().bind(...)` concurrently with running executables | Staging a new version compiles while the old one serves. That touches the module-global compile caches. |
| Registry `resolve` / `lazy` | Staging loads documents, which can import and register types while requests run. |
| `boundary` extraction, `raw_str` codes, params validation and conversion | They are reached from `run` / `score`. |

| Not thread-safe (documented, not fixed) | Why |
|---|---|
| `Session` | One debugger drives one session. It may be opened *from* a shared `Executable` without disturbing it. |
| `TraceSink` | One sink per call. Share a sink between concurrent calls and its counters race. If that's ever needed, give each call its own sink and merge them. |
| `Experiment`, `checks.run`, `diff`, `data` loaders, the bridge, MCP | Single-caller tools. Experiments isolate revisions in subprocesses already. |
| The step functions users write | Decider can't make them safe. The `impure` / `mutates_input` checks (DCD002/003) say why it matters: "this runs concurrently when served". |

Docstrings say this in one line:
- `Executable`: "Safe to call from many threads at once."
- `Session` / `TraceSink`: "Not thread-safe: use one per thread / per call."

## Rules for code on the serving path

1. **An `Executable` and its runner are immutable once compiled.** Everything a call needs that varies per call travels in the call: the `RunParams` (already per call), the `State`, the trace sink and its table. Nothing per call is ever stored on `self`.
2. **Compile once, publish once.** A runner compiles into locals under its own lock, double-checked on the plan. It then publishes one immutable snapshot with a single assignment (`self._compiled = Compiled(units, reads, strs, converted, plan)`). A call reads `self._compiled` once and uses that object throughout, so it never sees attributes from two different compiles.
3. **Caches are write-once, and entries carry their own keep-alives.** Insert with `setdefault` and never overwrite. An entry that holds raw pointers holds the buffers they point into, in the same immutable object.
4. **Reusable scratch is pooled per thread.** Pop it while in use, so a re-entrant call gets its own. `boundary/extract.py`'s `FrameView` pool already does this correctly. Move it into a shared helper so the planned per-(plan, thread) `State` reuse (`serving-latency.md`) follows it too.
5. **Module-global build caches use a double-checked lock.** The fast path is a lock-free `dict.get`. The miss path takes the cache's lock, checks again, builds, and publishes. Any object is fully set up before it is published, so `dispatcher._cache = ...` happens before the dispatcher is visible.
6. **Don't rely on the GIL.** Decider runs on CPython 3.14 with the GIL today (`Py_GIL_DISABLED=0`). Rules 1–5 hold on a free-threaded build too, because they never depend on "dict ops are atomic under the GIL". Rule 3's `setdefault` relies only on per-object dict atomicity, which free-threaded CPython keeps.

Rules 2, 3 and 5 put locks only on miss paths. A warm `score()` takes no lock, so the latency figures in `serving-latency.md` don't move.

## Fix list

The first four rows could corrupt results or crash. The rest only waste work or lose updates.

| # | Finding | Where | Fix | What users see change |
|---|---|---|---|---|
| 1 | **Use-after-free of `str` param spans.** `_alive[key]` and `_converted[key]` are written separately, so a concurrent call on the same key overwrites `_alive` and the first call's buffer is freed under its kernel. | `stepped.py:181-194` | One immutable entry `(bundle, utf8)`, inserted with `setdefault` and never overwritten (rule 3). The dict lives as long as the compiled snapshot. | nothing |
| 2 | **First-call compile race.** Two concurrent first calls both `_compile`. The second rebinds `_converted, _alive = {}, {}` (`stepped.py:100`) while the first's kernel reads spans from the old `_alive`. That is the same use-after-free, on cold executables. Attributes are also published one by one, so a call can mix them. Staged handlers are warmed by `stage()`, so this hits embedders and tests rather than `decider serve`. | `stepped.py:63-101`, `fused.py:37-40` | Rule 2: lock, double-check, one snapshot. | nothing; first-call latency unchanged |
| 3 | **NEW: the trace sink is stored on the shared runner.** `iterate` sets `self._trace = trace` and resets it to `None` in `finally`. A traced call and an untraced call that overlap send events to the wrong sink, or drop them mid-run. | `interpreted.py:73-84` | Rule 1: the sink and its table travel on the per-call `RunParams` (or a small call context) passed to `_node`. | nothing |
| 4 | **`exe.report` is last-writer-wins**, and `_checked` / `_warned` / `cache` are mutated per call. | `engine.py:102-113, 148-169` | `exe.report` becomes a property over a `threading.local`, meaning "the latest call **on this thread**". That is the same value for every single-threaded reader (tests, MCP `tools.py:204`, notebooks), and correct per thread in serving. `_checked` / `_warned` go through the write-once helper, so each `FallbackWarning` fires once. | nothing single-threaded; per-thread reports in serving |
| 5 | **`ParamsCache._results` read-modify-write.** Validation is duplicated and report updates are lost. | `params/validate.py` `ParamsCache.validate` | `setdefault`; the `_last` slot is already race-tolerant | nothing |
| 6 | **Module-global compile caches are check-then-act**, plus an unlocked `dispatcher._cache = _SaltedCache(fn)` on a shared dispatcher. | `njit.py:43-44, 83-89, 146-152`; `kernel.py:95, 117-125`; `structs.py:23, 44-51`; `rows.py:38, 41-49` | Rule 5: one `once(cache, key, build)` helper with a lock per cache. The salted cache is attached inside `build`, before the dispatcher is published. | nothing |
| 7 | **The fused runner mutates `packed` in place** when a branch falls back after `ArmOutOfRange`. | `fused.py:79` | Copy-on-write: `self._compiled = snapshot._replace(packed=without_path)`, a single swap that readers see as either old or new. | nothing |
| 8 | **NEW: `Session` shallow-copies the shared runner** (`session.py:83`, `edit.py:177`), so it shares the runner's caches with the serving executable. It is safe today only because `_compile` rebinds the dicts. | `engine/debug/session.py:83`, `edit.py:174-210` | After rules 1–3 the shared parts are immutable snapshots, so a shallow copy is correct by construction. Add one test that opens a session on a serving executable mid-load. `visit` / `skip` are only ever set on the copy; keep it that way. | nothing |
| 9 | **The registry is mutated at runtime** by `resolve()` imports and `lazy()`. | `registry/core.py` | `resolve` / `lazy` take the root's lock on the miss path | nothing |
| 10 | `Live.dates: dict = {}` is a shared mutable default | `serving/handler.py:33` | `MappingProxyType({})` | nothing |
| 11 | `_BatchRunner`'s lazy `bind` (old audit, item 4) | `steps/each.py` | **Gone.** This branch's `each` rework moved it into the IR as `ScatterGather`. The stress test covers `each` so it stays gone. | |
| — | Already correct: `_RAW_STR_CODES` double-checked lock (`types.py:12`); `extract.py` `lru_cache` + per-thread pool; read-only sentinels `InterpretedRunner.skip` and `_NO_FRAME`; `RequestHandler` swap lock | | keep | |

**A NEW item that is a question, not a bug.** `RequestHandler.process_fn` calls `score()` / `run()` directly on the event loop (`handler.py:203-210`). With Starlette, one worker process therefore serves requests one at a time, and the `nogil` kernels never overlap inside it. Concurrency today comes from embedders calling one `Executable` from their own threads, and from many processes. The contract above is needed either way. Whether `process_fn` should move scoring to a thread pool (`anyio.to_thread.run_sync`) is a latency trade-off: a thread hop on every request in exchange for not blocking the loop. It should be measured in `benchmarks/` before anyone decides. It isn't part of this fix.

## Keeping it fixed

1. **A stress test, `tests/serving/test_concurrency.py`.**
   - It runs `ThreadPoolExecutor(16)` with `sys.setswitchinterval(1e-6)` to force interleavings, repeated a few hundred times, for every mode. Every result is compared with the serial answer.
   - Cases:
     - (a) a cold `Executable`, with concurrent first calls (#2);
     - (b) `str` params on the span path, alternating two params documents (#1);
     - (c) half the calls traced, each with its own sink, where each sink must get exactly its own call's events (#3);
     - (d) `exe.report` per thread matches that thread's document (#4);
     - (e) `stage` + `activate` while requests run;
     - (f) a packed branch hitting `ArmOutOfRange` mid-load (#7);
     - (g) `each`;
     - (h) a `Session` opened on the shared executable mid-load (#8).
   - Races are probabilistic, so a failure here is a real bug, while a pass is only evidence. The case (a)/(b) workload is the one that would have caught #1 and #2.
2. **A convention test in `tests/test_conventions.py`.** It does an AST pass over the runner classes and `Executable` in `_engine/run/` and fails on any assignment or in-place mutation of `self.<attr>` (`=`, `[...] =`, `.update`, `.pop`, `.append`, `.add`). The only exceptions are `__init__` and the lock-held compile method, plus an allowlist with a reason. This is the guard that would have stopped #3 from landing.
3. **Docstrings** state the contract as in the table above.

## Where it goes in the plan

- `_engine/concurrency.py` (new) holds `once(cache, key, build)` and the per-thread pool, moved from `boundary/extract.py`. It is used by the runners, compile caches, params and registry. It is private, and there's nothing for users to learn.
- **Phase 0 (bugs)** gets fixes #1–#4 and #7, together with the stress test. They are memory-safety and correctness fixes, and they're easier to review **before** `engine` → `_engine` moves the files. #5, #6 and #9–#10 can land in the same phase or ride along with phase 3, since they are small and mechanical.
- **Phase 1 (guard rails)** gets the convention test.
