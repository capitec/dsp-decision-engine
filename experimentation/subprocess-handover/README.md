# Experiment K — subprocess compile and cache survival, together

Tests doc 08 §4's `live`-mode lifecycle end to end — the part EXPERIMENTS.md §H
and §C each tested half of, but never together:

```
child compiles -> pinned cache dir -> parent loads with ZERO compilations
-> atomic swap -> serving continues
```

## What this measures

1. **M1+M3** (`phase_m1_m3`) — a parent serves a 10-rule kernel continuously
   while a **subprocess** compiles a new 30-rule generation into a pinned
   dir; serving throughput is measured *during* the compile (as §H measured
   the thread case), and after the child exits, the **parent loads the new
   generation in-process** with `numba.core.event.install_recorder` wrapped
   around the call, so a genuine numba compile in the parent — not just "no
   exception" — is what "zero compilations" means here.
2. **M2** (`phase_m2`) — do numba's six cache-survival conditions
   (EXPERIMENTS.md §C) survive a real child/parent process boundary? Forces
   child and parent to disagree, one axis at a time, on: cwd (relative-path
   resolution), `NUMBA_CACHE_DIR`, the generated filename, and the `def`
   line number (isolated from mtime/size by construction — see
   `handover.py:phase_m2`'s comment). Found a **seventh condition** by
   accident: `sys.modules` registration name.
3. **M4** (`phase_m4`) + **M4c** (`phase_m4c_pyc_confound`) — the stale-code
   hazard from §C, reproduced *across* the child/parent boundary, plus a
   root-cause correction (see Findings, #2 below).
4. **M5** (`phase_m5`) — does content-addressed naming
   (`drv_<sha256[:16]>.py`) fix the M4 hazard without destroying cache hits
   for genuinely unchanged content?
5. **M6** (`phase_m6`) — wall clock from "config change arrives" to "new
   generation serving", for a 10-rule and a 30-rule change, using the M5 fix.

## Reused, not rewritten

- `emit.emit_ruleset` / `emit.count_lines` — `experimentation/ruleset-compile-latency/emit.py`
- `compile_kernel`, `serve_until`, `pct`, `med`, `make_args`, `build_src`,
  `count_compiles`'s `numba.core.event.install_recorder` pattern, `rss_mb`,
  `ROWS` — imported live from `experimentation/staged-compile-atomic-swap/run.py`
  (not copy-pasted; see `handover.py`'s `_import_module_from_path`)
- `gen_driver.gen_source` (a driver purpose-built for cache-key tests) and
  the subprocess-JSON-line child pattern —
  `experimentation/numba_cache_survival/gen_driver.py` and `child.py`.
  `child_op.py` in this directory *is* that pattern, extended with
  `numba.core.event` compile-event counting, which `child.py` did not have.
- the same-byte-length constant-edit trick (`"* 0.5"` → `"* 0.7"/"0.9"` on
  the `a1` line) — `numba_cache_survival/run_experiment.py:part_stale`

## Run

```
.venv/bin/python handover.py            # all measurements, ~40s
.venv/bin/python handover.py --phase m2 # one phase; choices: m1m3 m2 m4 m4c m5 m6
```

Results append to `results.jsonl`, one JSON record per measurement, flushed
and `fsync`'d immediately after each write.

## Memory

Every array here is `ROWS=100_000` rows × a handful of float64/int64/bool
columns — the harness prints the estimate (**17.9 MB**) before running
anything. Measured **peak RSS: 176 MB**. `free -g` was checked before
starting (23 GB available, run proceeded) and is checked automatically on
every invocation; it refuses to start under 8 GB available. This never
approached the 2 GB threshold, so it was **not** run under
tmux/systemd-run — that machinery is for runs *expected* to cross ~2 GB, and
using it here would only add overhead. Total wall clock for the full run:
**42.4 s**, comfortably inside the 12-minute budget (nothing was dropped).

## Findings

Quoted claims are tested verbatim against what was measured. **Two
confirmations, one solid refinement of an existing "confirmed" finding, and
one new hazard not in any doc.**

### 1. CONFIRMED — doc 08 §4.1's chain holds, and better than §H's own subprocess number suggested

> §H: "A subprocess compile of a 30-rule kernel measured 5.36 s with serving
> untouched... the parent then loads from the pinned cache directory" —
> untested until now.

Measured together: child subprocess compile of a 30-rule kernel, **5.26 s**
(consistent with §H's 5.36 s). **Serving throughput during that compile:
97.9% retained** (baseline median 1.129 ms/call vs. 1.154 ms/call during —
essentially noise; p95 1.239 ms, one 5.1 ms outlier out of 5,031 calls,
almost certainly process-spawn scheduling jitter, not a coupling effect).
This is dramatically better than §H's *thread*-based numbers (26–55%
retained) — exactly the point of doc 08 §4's "must be a subprocess" call,
now verified with a **process**, not a thread, in the loop.

**The parent load registered `parent_compile_events_on_load: 0`** via
`numba.core.event.install_recorder("numba:compile")` wrapped directly around
the parent's `njit(cache=True)(mod.kern); disp(*args)` call — a real
assertion, not an absence of exceptions. Parent and child computed
**bit-identical checksums** (41,951,207 both sides) on the 100k-row output,
so this isn't just "zero compiles", it's "zero compiles and correct
handover". **Doc 08 §4.1's chain, as a mechanism, works.**

### 2. REFINED — §C's "numba serves stale compiled code" is CPython's bytecode cache, not numba's

> §C: "Change a numeric constant in a generated driver, keep the file size
> identical, restore the mtime: numba reports a cache HIT and returns the
> pre-edit answer... The index key hashes `co_code`, which excludes
> `co_consts`."

**The symptom reproduces, exactly as described** — I reran
`numba_cache_survival/run_experiment.py`'s own unmodified `part_stale()` and
got the same stale HIT (`1.25575` served, true value `1.15565`) it always
has. But building this harness's `child_op.py`, every subprocess-boundary
load of an edited-in-place, mtime/size-preserved generated file came back
**correct**, not stale — which shouldn't have been possible if numba's own
cache were the cause. The difference: `child_op.py` sets
`sys.dont_write_bytecode = True` and deletes any stray `.pyc` before
importing (added for an unrelated reason — see finding 3 — and it
accidentally fixed this too).

**M4c isolates it with a controlled A/B.** Same on-disk state both times
(same edited `drv.py`, same `mtime`/`size`, numba's own `.nbi`/`.nbc` files
*completely untouched* between the two branches) — the *only* difference is
whether `__pycache__/drv.cpython-314.pyc` (CPython's **own** bytecode
cache, also keyed on `(mtime, size)`, written automatically by
`spec.loader.exec_module()` unless bytecode-writing is disabled) is present:

| branch | numba cache touched? | value served | correct? |
|---|---|---|---|
| `.pyc` present (unmodified, as §C's harness runs) | no | 1.25575 (gen A) | **stale** |
| `.pyc` deleted, numba's `.nbi`/`.nbc` identical | no | 1.15565 (gen B) | correct |

`STALE_CAUSED_BY_CPYTHON_PYC: True`. The mechanism: `spec.loader.exec_module()`
validates the `.pyc` against the *current* file's `(mtime, size)` — which we
preserved — finds a match, and hands back the **old** compiled Python
function object without ever re-parsing the edited source. Numba's
decorator then runs on *that* (unchanged) function object, so of course its
own cache looks consistent — it never saw a change. Separately, in every
edit I could construct where the *only* thing changing was a source-level
numeric literal, `co_code` (via `dis`-level inspection) **did** change too
— CPython's constant pool is a deduplicated-by-value list built in
first-appearance order, so swapping one literal's value routinely
renumbers or appends a slot, which changes the `LOAD_CONST` operand at
that site, which changes `co_code`. I could not build a case where
`co_code` stayed byte-identical after a semantically real edit, so I
cannot confirm numba's own index-key mechanism is *itself* exploitable via
an ordinary literal edit in this Python/numba version — only that
**CPython's own bytecode cache reliably is**, and it sits one layer in
front of numba's, so a fix aimed only at numba's cache (byte-identical
codegen, pinned paths) is not sufficient on its own.

**Practical upshot, unchanged from §C's bottom line**: a build that
normalises mtimes for reproducibility (doc 05 §4.2's own stated
requirement) *does* convert a stale-constant edit into a silently wrong
answer — just via `__pycache__/*.pyc`, not `.nbi`/`.nbc`. The fix is the
same either way (`sys.dont_write_bytecode = True` for the generated-driver
directory, or content-addressed naming — see finding 4 — which sidesteps
both caches structurally rather than relying on a flag).

### 3. NEW — a seventh cache-survival condition: `sys.modules` registration name

Not in §C's six, because that harness's own `child.py` always registers
(`register_in_sys_modules` defaults `True`) and never varied it against a
*different* loading process. Building M1+M3's parent-load code without this
produced:

```
ModuleNotFoundError: No module named '<dynamic>'
```

raised **deep inside `numba/core/environment.py:_rebuild_env`, from
`pickle.loads`**, not at import time and not anywhere near anything that
looks like caching. numba pickles the compiled `Environment` by the
module's registered name (via `inspect.getmodule`); if the compiling
process never put the module in `sys.modules`, or the loading process uses
a *different* registration name, the cache entry is **permanently
unloadable by any process, including a second load in the same one** — not
"less efficient", genuinely broken. Confirmed with the dedicated M2 axis
test: unregistered load → `ModuleNotFoundError: No module named
'decider2_gen_driver_A'`; wrong-name load → the same error under the wrong
name; matching-name load → clean `HIT`. **Verdict: loud, but cryptic** —
nothing in the traceback mentions caching, modules, or registration, so an
engineer hitting this in production would not connect it to "the
subprocess compiler and the parent loader must agree on a registration
name" without already knowing to look for it.

**Doc 08 §4.3's "pinned, stable directory" contract needs one more clause**:
the module registration name is part of the contract too, and it must be
derived deterministically and identically by both the compiling subprocess
and every future loader (e.g. from the generated filename itself, which
content-addressing — finding 4 — already makes a good key for).

### 4. CONFIRMED — content-addressed naming fixes the hazard without destroying legitimate hits

> Proposed here (not in any existing doc): name generated files
> `drv_<sha256(source)[:16]>.py`, and skip the write if that path already
> exists.

Two sub-cases, both measured:

- **Unchanged content across two independent "redeploys"**: same hash →
  same path → second write is skipped entirely (file, and therefore its
  `mtime`, is never touched) → second load is a genuine `HIT`, 0 compile
  events. Cache survives a redeploy **honestly**, not by mtime luck.
- **Changed content** (M4's exact edit): different hash → **structurally
  different path** → the file the loader reads is brand new, `MISS`,
  fresh correct compile (`1.05555`, matches ground truth). There is no
  `(mtime, size)` pair left to coincide on, by construction — this closes
  the hazard at the naming layer, which also closes CPython's `.pyc`
  version of it for free (a different path means a different, and in this
  case nonexistent, `.pyc`).

**This is the one clean fix for both layers 2 and 3 above**: pair
content-addressed filenames with a registration name *derived from the same
hash*, and the mtime/size/lineno/registration-name conditions stop being
things a deploy tool has to get right, because they're never ambiguous.

### 5. Six-condition survival across the process boundary (M2)

| axis (child vs. parent disagree on) | outcome |
|---|---|
| generated filename | **loud**: `FileNotFoundError`, immediate |
| cwd (relative-path resolution) | **loud**: `FileNotFoundError`, immediate, *if* the mismatch resolves to nothing; a coincidental same-named file in the other cwd would instead be a silent, independent, correct `MISS` (not tested directly — inferred from condition 1's known behaviour in §C) |
| `NUMBA_CACHE_DIR` | **silent MISS, but safe**: recompiles into the (now-empty) directory the parent is actually looking in; never observed serving the wrong dir's stale entry |
| `def` line number (isolated from mtime/size, which were held frozen) | **silent MISS, safe**: the line-number component of the cache key is real, independent protection — it forced a correct recompile even though `(mtime, size)` matched exactly |
| `sys.modules` registration name | **loud, but cryptic** (finding 3) |

**None of the four axes tested alone produced a silent wrong-answer HIT.**
The only mechanism that did, across the whole experiment, was the
`(mtime, size)` coincidence from finding 2 (and its CPython, not numba,
root cause) — which is exactly the scenario doc 05 §4.2's own reproducible-
build requirement invites.

### 6. Wall clock, "config change arrives" → "serving" (M6, with the content-addressed fix)

| rules | child compile | child process wall | parent load | parent compile events | total |
|---|---|---|---|---|---|
| 10 | 1.91 s | 2.54 s | 5.7 ms | 0 | **2.56 s** |
| 30 | 6.66 s | 7.31 s | 7.2 ms | 0 | **7.34 s** |

Consistent with §H's own 5.36 s/6.14 s for a 30-rule subprocess compile (this
run's 6.66 s/7.31 s is the same order, plausibly seed/machine-load
variance — not re-measured for variance here, see Dropped below). Atomic
swap itself is not re-measured in this harness; §H already measured it at
0.177 µs median and it is cited, not re-run, in the `total` column (its
contribution is negligible at 4+ orders of magnitude below the compile
time). **Parent compile events were 0 in both rows** — the M1 assertion
holds at both rule counts, not just 30.

## What a configuration UI can honestly promise

**Doc 08 §4's lifecycle works as written, mechanically** — subprocess
compile leaves serving at ~98% throughput, the parent genuinely loads with
zero compilations, and the two processes agree on a result. **But the
"pinned cache directory" contract in §4.3 is underspecified**: it needs (a)
content-addressed generated filenames, not merely "stable" ones, or the
exact mtime-preserving-rebuild scenario doc 05 §4.2 asks for as a build
practice silently serves a stale generation — one layer up in CPython's own
`__pycache__`, not primarily in numba's cache as previously attributed; and
(b) a `sys.modules` registration name derived the same way by every
compiler and every loader, or the failure mode is a `ModuleNotFoundError`
that gives no hint it's a caching problem. With both of those in place, a
UI can honestly promise: **~2.5 s for a 10-rule change, ~7.5 s for a
30-rule change**, measured end-to-end from source-on-disk to a verified,
checksum-matching, zero-compile parent load — consistent with, and now
load-bearing evidence for, G's "~10 s up to ~35 rules" ceiling.

## Dropped, for the time budget

- Variance/repeats on M6 (single run per rule count — the numbers are
  directionally consistent with §H/§G, not statistically characterised
  here).
- The "coincidental same relative filename in a different cwd" sub-case of
  the cwd axis (would need a second fixture directory that happens to
  contain an unrelated `drv.py`; inferred instead from §C's already-measured
  "identical bytes, different dir → MISS").
- Re-measuring the atomic swap latency (cited from §H instead — it is 4+
  orders of magnitude below everything else measured here, so re-measuring
  it would not change any conclusion).
- Testing the `magic_tuple` (CPU target) condition across the boundary —
  this machine only has the one target; would need to fake
  `NUMBA_CPU_NAME`, which §C already exercised in-process.
