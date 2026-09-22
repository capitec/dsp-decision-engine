# Lazy string matching in a numba kernel via C (PCRE2) — results

*Experiment directory: `experimentation/numba-c-string-matching/`. Sibling strand: `experimentation/rust-string-matching/` (Rust). Every number below is in `results.jsonl`; `./run_all.sh` reproduces them in ~8 minutes.*

## Recommendation, and the number that drives it

**Do not add a C regex dependency to decider2 for speed. Do adopt the typed-feature split, for correctness.**

The number: on the owner's own shape (`ft1 > c1 AND ft2 > c2 AND ft3 ~ /^dog/`) with **no rows reaching the regex node**, evaluating the regex lazily in-kernel through C costs **15–24 ms per million rows** against **33–62 ms** for today's route (polars `str.contains` over every row before the kernel). That is a 2.5–3.5× win on the string node — and it amounts to **20–40 nanoseconds per row**. For a 10,000-row scoring batch it is 0.2–0.4 ms. The mechanism works, caches, and is guarded, but the total stake is too small to justify a C build step, a backtracking regex engine, and a raw-pointer surface in a bank's credit path.

The typed-feature split is a different matter: today every feature is coerced to one `float64` array, and a scaled-int64 money value above 2⁵³ compares **wrongly** (two distinct int64 values compare equal; `9007199254740993 > 9007199254740992` evaluates false). Separate typed arrays fix that outright, cost **5–20% on the walk** (1.5–8 ns/row), and are what would give any string matcher — C, Rust, or none — the raw bytes to work on. That change stands on its own.

If in-kernel regex is wanted as a *capability* (rule authors writing `match_type='prefix'|'regex'` without a frame-tier prep step) rather than for speed, this strand shows numba+C can do it honestly — see §7 for the residual hazards, one of which (PCRE2 is a backtracking engine) argues for the Rust strand's engine over this one.

## 1. The problem, for someone who has not followed the thread

decider2 compiles decision trees to numba kernels. Strings cannot be used inside a numba kernel (numba has no `re` in nopython mode), so at the boundary every string column is dictionary-encoded into an `int32` code, and a tree node can only test a string for **exact** equality against a code. Prefix, suffix, substring and regex tests raise `UnsupportedInKernel`; the documented workaround is to compute the test in polars beforehand (`pl.col(...).str.contains(...)`) and branch on the resulting bool column. That is the "frame tier" — the incumbent, with no dependencies and nothing to crash.

The owner's objection: the frame tier does O(rows) work always, while decider2's AND short-circuits, so in `ft1 > 5 AND ft2 > 10 AND ft3 ~ /^dog/` where no row passes `ft1 > 5`, a *lazy* regex at the node runs **zero** times. Prior experiments established the raw mechanism: an njit kernel can call C through ctypes at ~3.84 ns/call (EXPERIMENTS.md §V), glibc's `regexec` works from njit but is slower than polars' Rust regex (§T: 60.3 ns vs 42.2 ns), and — the thing that killed this twice — a kernel that captures a ctypes symbol as a global can never be disk-cached, whereas a pointer passed as an *argument* caches (§W).

This strand asks: with a better C regex (PCRE2), pointers passed as arguments, and typed feature arrays, does lazy in-kernel matching beat the frame tier, by how much, and is it safe?

## 2. What was built

| File | What it is |
|---|---|
| `c/strmatch.c`, `c/build.sh` | 230-line C shim over the system `libpcre2-8.so.0` (JIT enabled) plus hand-rolled `exact`/`prefix`/`suffix`/`substring` matchers and a glibc `regexec` control. Three entry points: `sm_compile → id`, `sm_match_id(id, ptr, len) → 1/0/<0`, `sm_free(id)`. **C owns the table of compiled patterns**; a kernel only ever passes a small integer id, which C bounds-checks before dereferencing anything. Builds in 0.2 s. |
| `strmatch.py` | ctypes binding. Exposes `MATCH_ADDR`, the raw address of `sm_match_id` as a Python int. Malformed patterns raise `PatternError` here, at build time. |
| `call_ptr.py` | Two `@intrinsic`s lifted from `cfunc-pointer-interpreter/call_ptr.py`: `call_match(addr, id, ptr, len)` (inttoptr + call) and `load_i64(addr)` (raw load). Nothing process-specific is baked into the compiled kernel. |
| `kernels.py` | Two walkers. `walk_f64` is today's shape (one float64 array, node kinds LEAF/CMP/IS_TRUE/IS_FALSE, verbatim in spirit from `decider2/src/decider2/trees/interpreter.py`). `walk_typed` is the proposal: separate `float64`/`int64`/`uint8`/`int32` arrays plus a `uint64[n_cols, 4]` pointer table over polars' Arrow `(offsets, values)` string buffers; each node carries `feat_kind` beside `feat_idx`; two new node kinds, `STR` (call C lazily) and `STR_MASK` (per-category lookup). Both `@njit(cache=True)`; the C address and pattern ids are **arguments**. |
| `bench_regex.py` | Per-call cost of each matcher from njit; the §V call-overhead probe. |
| `bench_selectivity.py` | The owner's shape at 0/1/50/100% selectivity × 2 cardinalities × 2 patterns, three strategies, identical answers asserted against a Python `re` oracle. |
| `bench_typed_walk.py` | Typed vs float64 walk on numeric-only trees; the int64 precision case. Run twice, with and without `NUMBA_BOUNDSCHECK=1`. |
| `ablate_refcount.py` | Why the first run was wrong (§6) — seven explicit kernel variants. |
| `cache_check.py`, `cache_worker.py` | Two separate processes, persistent cache dir, `NUMBA_DEBUG_CACHE=1`. |
| `safety_check.py` | Eleven failure cases, each in its own subprocess, exit code recorded. |
| `results.jsonl` | Every measurement, appended as made. `results_run1_polluted.jsonl` is the invalid first run, kept as a warning (§6). |

Environment: Python 3.14.5, numba 0.67.0, polars 1.41.2, numpy 2.4.6, system PCRE2 10.47 (`libpcre2-8.so.0.15.0`), GCC 16.1. Single-threaded throughout; 28-core shared box.

## 3. How fast is C regex from njit? (per call, over Arrow buffers)

200k rows, ns per call, loop floor 0.4 ns subtracted where marked. "low" = 12 distinct short words; "high" = one distinct value per row (`dogma-0001234`).

| Matcher (called from njit through the pointer table) | `^dog` low | `^dog` high | `^d[o0]g\w*(ma\|e\|s)$` low | … high |
|---|---|---|---|---|
| **PCRE2 JIT (UTF mode)** | **33.6** | **34.1** | **40.4** | **29.7** |
| PCRE2 JIT, byte mode | 33.6 | 28.5 | 41.0 | 29.6 |
| PCRE2 interpreter (no JIT) | 78.7 | 73.8 | 108.4 | 102.7 |
| glibc POSIX `regexec` (the §T control, same harness) | 85.1 | 70.6 | 156.0 | 247.5 |
| hand-rolled literal prefix (`memcmp`) | **14.1** | **9.4** | — | — |
| polars `str.contains`, whole column, for reference | 45.9 | 24.4 | 39.0 | 24.9 |
| Python `re` loop, for scale | 622 | 641 | | |

Read: **PCRE2-JIT from njit is 29–40 ns/call — 2–7× faster than glibc and level with polars' Rust regex** (which is 24–46 ns/row here, and 25.6 ns pure-compute in §U). The §T conclusion that libc is the weak part was right; PCRE2 removes it. The registry indirection (id → pointer, bounds-checked in C) costs nothing measurable (33.6 vs 33.1 ns raw-pointer). And for the owner's actual pattern, `^dog` is a *prefix test*: the 10-line `memcmp` matcher does it in 9–14 ns, a third of any regex engine. A `match_type='prefix'` node should never touch a regex engine at all.

Call overhead, §V's `(int32,int32)->int32` shape: **1.64 ns/call** through the pointer-table argument, 2.16 ns through a captured ctypes global (the uncacheable mechanism). Consistent with §V's 2.4–3.84 ns.

## 4. The measurement that matters: the owner's shape, by selectivity and cardinality

1,000,000 rows, ms per batch, **total = each strategy's own precompute + its walk**, min of 5. Every cell's output was asserted identical to the others and to a Python `re` oracle. `c2` is set so `ft2 > c2` is always true (the node is walked but never filters); `c1` sets the fraction reaching the regex node.

Per-batch precompute each strategy pays regardless of selectivity:

| | low card (12 distinct) | high card (10⁶ distinct) |
|---|---|---|
| frame: polars `str.contains` over every row | 47–52 ms | 27–31 ms |
| category: dictionary-encode the column | 33–34 ms | 85–92 ms |
| category: regex over the distinct values | **0.2 ms** | 35–40 ms |
| lazy: `_get_buffers()` to get Arrow offsets+values | 13 ms | 6–7 ms |

**`^dog`, totals (ms per 1M rows):**

| rows reaching regex | today (f64 walk + frame) | frame (typed walk) | per-category, codes already paid / incl. encode | **lazy C at node** |
|---|---|---|---|---|
| low card, 0% | 58.6 | 61.8 | **10.4** / 43.3 | 23.5 |
| low card, 1% | 59.4 | 62.3 | **12.0** / 44.8 | 26.1 |
| low card, 50% | 76.7 | 83.3 | **32.3** / 65.2 | 62.1 |
| low card, 100% | 69.0 | 80.5 | **30.7** / 63.6 | 78.5 |
| high card, 0% | 45.1 | 41.7 | 50.5 / 142.1 | **17.8** |
| high card, 1% | 39.0 | 42.1 | 52.2 / 143.7 | **19.1** |
| high card, 50% | **55.8** | 61.4 | 72.0 / 163.5 | 55.9 |
| high card, 100% | **46.2** | 49.7 | 60.5 / 152.0 | 62.4 |

The harder pattern `^d[o0]g\w*(ma|e|s)$` gives the same shape (lazy 15–25 ms at 0–1%, 50–86 ms at 100%; see `results.jsonl`).

What the table says:

1. **Lazy wins exactly where the owner said it would, by 2.5–3.5×, and loses by 10–35% at 100%.** The crossover is around 50% of rows reaching the node. The win is entirely the frame precompute avoided (27–52 ms) minus the buffer fetch (6–13 ms); the walk itself is identical at 0% (10 ms) and 65–73 ms at 100% (the walk plus ~35–55 ns of PCRE2 per row).
2. **Per-category is the best low-cardinality answer by a wide margin** — 10–32 ms — *if* the column is already dictionary-encoded, which in decider2 today it always is. It collapses at high cardinality (142–164 ms: encoding a million distinct strings costs more than any regex). §T's conclusion stands: per-category by default, switch on a measured cardinality threshold.
3. **`_get_buffers()` is not zero-copy** on polars 1.41. Strings are stored in the binview layout internally and the call materialises offsets+values: 11 ns/row, O(n) (1.1 ms / 10.9 ms / 43.6 ms at 10⁵ / 10⁶ / 4·10⁶ rows; two calls return different memory). So the lazy path is not O(rows reaching) — it is O(rows)·11 ns + O(reaching)·35 ns. Still cheaper than polars' O(rows)·30–50 ns, but the "nothing to pack" premise is a third of the story. (`to_arrow()` might expose the view layout directly; pyarrow is not installed so it is untested. Reading binview in-kernel is possible — 16 bytes per row, inline for ≤12 bytes — and would make the lazy path truly zero-copy.)
4. **Everything is tens of milliseconds per million rows.** The whole string node, on the slowest strategy, is under 60 ns/row.

## 5. Typed arrays vs today's single float64 array

Numeric-only random full binary trees (4 float64, 2 int64, 2 bool features), 1M rows, both walkers encode the *same* tree, answers asserted identical:

| depth | nodes | f64 walk ns/row | typed walk ns/row | ratio | with `NUMBA_BOUNDSCHECK=1`: f64 / typed |
|---|---|---|---|---|---|
| 1 | 3 | 10.3 | 11.8 | 1.14 | 11.7 / 14.2 |
| 3 | 15 | 25.2 | 29.6 | 1.17 | 28.1 / 36.6 |
| 5 | 63 | 42.9 | 51.4 | 1.20 | 49.4 / 60.3 |
| 8 | 511 | 79.5 | 83.3 | 1.05 | 72.1 / 94.3 |

The `feat_kind` switch costs **1.5–8 ns/row (5–20%)**. On the owner's 3-node shape in §4 the gap was wider in places (e.g. 17 vs 29 ms at 100%) — run-to-run noise on a shared box is of that order; the numeric benchmark is the controlled one.

**int64 precision, demonstrated** (`section: int64_precision`): values `[2⁵³+1, 2⁵³, 2⁵³−1, 2⁵³+1]` against threshold `2⁵³`:

| op | truth | today's f64 walk | typed walk |
|---|---|---|---|
| `==` | `[0,1,0,0]` | `[1,1,0,1]` **wrong** | `[0,1,0,0]` correct |
| `>` | `[1,0,0,1]` | `[0,0,0,0]` **wrong** | `[1,0,0,1]` correct |

2⁵³ cents is R90 trillion, so this is not a practical money bug today — but doc 03 §1.2's scaled-int64 rule promises exactness, the kernel silently breaks the promise, and an int64 *identifier* (account/customer number) above 2⁵³ is entirely realistic. The fix is the typed split; it needs no C.

## 6. The decisive non-performance checks

### 6.1 Disk caching — demonstrated across two processes

`cache_check.py` wipes a persistent `NUMBA_CACHE_DIR`, runs `cache_worker.py` twice as separate processes with `NUMBA_DEBUG_CACHE=1`, and counts numba's own log lines:

| run | `[cache] data saved` | `[cache] data loaded` | typed kernel first call | wall |
|---|---|---|---|---|
| cold | **2** | 0 | 1134 ms | 2.28 s |
| warm | **0** | **2** | 178 ms | 0.94 s |

Both kernels (`walk_typed`, the one that calls C, and `walk_f64`) hit; `.nbi`/`.nbc` files are in `.numba_cache_persist/`. In the same warm process a contrast kernel that captures the ctypes symbol as a global with `cache=True` produced numba's `Cannot cache compiled function "global_capture" as it uses dynamic globals (such as ctypes pointers …)` warning. §W's rule holds exactly: **pointer as argument caches; symbol as global does not.** Logs: `cache_cold.log`, `cache_warm.log`.

### 6.2 Malformed input — demonstrated, each case in its own subprocess

| case | outcome |
|---|---|
| malformed regex `^(dog` | `PatternError: rejected at offset 5: missing closing parenthesis` at compile time, in Python; never reaches a kernel |
| pattern slot out of range, validated program | `ValueError: node 2: pattern slot 7 out of range (1 handles)` at build time |
| pattern slot out of range, **unvalidated** | kernel guard: 1000/1000 rows return `ERR_LEAF` (−1); no crash |
| string column index out of range, unvalidated | kernel guard: 1000/1000 rows `ERR_LEAF` |
| corrupted offsets pointing 10⁹ bytes past the buffer | kernel guard: exactly the 501 corrupted rows `ERR_LEAF`, the rest correct |
| null id (0) with an empty registry; garbage id `0xDEADBEEF`; ids −1, 2⁶², 4095 | C bounds-checks against its registry: all rows `ERR_LEAF` |
| freed pattern reused | `ERR_LEAF` (magic zeroed on free — best-effort, not a guarantee) |
| catastrophic backtracking `^(a+)+$` on `a…ab` | PCRE2 match-limit hit → `ERR_LEAF`, **211 µs per row** at the limit of 50,000 set in the shim; at PCRE2's default limit it was **42 ms per row** |
| numeric `feat_idx` out of range, unvalidated | **SIGSEGV** — numba bounds checking is off, as in decider2 today. `Program.validate()` catches it at build. decider2's real walker indexes a per-row *tuple*, and numba raises `IndexError` for an out-of-range tuple index, so decider2 is safer here than this prototype's 2D arrays. `NUMBA_BOUNDSCHECK=1` costs 15–30% (§5). |

An earlier version of the shim handed raw pointers to the kernel; a made-up pointer segfaulted. Moving the pattern table into C and passing only ids closed that at zero cost. **The remaining C-side hazard is intrinsic to the engine, not the binding**: PCRE2 backtracks, so a rule author can write a pattern that costs 0.2 ms per row even with the limit in place. polars' and Rust's `regex` are linear-time by construction and cannot. In a credit path where rules are authored by people, that is a real argument for the Rust strand's engine if lazy matching is adopted at all.

### 6.3 The trap that invalidated the first run — worth knowing for any typed walker

The first full run showed every typed strategy at a flat **~420 ms per 1M rows** regardless of selectivity or tree depth (kept in `results_run1_polluted.jsonl`). The cause, found by ablation (`ablate_refcount.py`, seven explicit kernel variants):

| variant | ns/row | NRT incref / decref call sites |
|---|---|---|
| single loop, numeric only | 23.1 | 0 / 29 |
| single loop + STR node, tuple of arrays | 20.0 | 20 / 60 |
| single loop + STR node, pointer table + raw loads | 20.8 | **0** / 35 |
| same body as a separate per-row `@njit` function, real call | **364.3** | 35 / 172 |
| same, `inline="always"` | **327.7** | 35 / 168 |

Written as a per-row function taking arrays — decider2's `walk_tree` shape — numba treats every array parameter as a fresh local and emits reference-count traffic for each of them on every row; the ~170 `NRT_decref` calls at ~2 ns each are the 365 ns. `walk_tree` in decider2 does not suffer because its per-row `feats` is a tuple of *scalars*. **A typed walker in decider2 must keep the string buffers as integers (`str_tab` of raw addresses, read through `load_i64`) and never pass arrays into the per-row function.** The final `kernels.py` does exactly that.

## 7. What I would do

1. **Land the typed-feature split** (`feat_kind` beside `feat_idx`, separate `float64`/`int64`/`uint8`/`int32` inputs, typed thresholds). It fixes a real correctness bug, costs 5–20% on the walk, needs no new dependency, and is the precondition for any string story. Keep per-row inputs as scalars or raw integers per §6.3.
2. **Keep the frame tier as the string mechanism.** It is 33–62 ms per million rows, has no build step, no crash surface, no cache risk, and is linear-time on any pattern. The lazy path's best case saves 20–40 ns/row. That is not a reason to ship C.
3. **If `match_type='prefix'|'suffix'|'substring'` is wanted in-kernel**, do it *without a regex engine*: the 10-line `memcmp`/`memmem` matchers are 9–26 ns, need 40 lines of C (or could be written in pure numba over the `uint8` buffer with no C at all — worth a two-hour probe), and have no pathological inputs. That would lift most of the `UnsupportedInKernel` restrictions at near-zero risk.
4. **If real regex-in-kernel is wanted as a capability**, this strand proves the numba+C route is viable and cacheable — but choose the engine on the backtracking axis, not speed (PCRE2-JIT and Rust `regex` are within noise of each other at 30–40 ns). Compare with the Rust strand on: linear-time guarantee, thread-safety (the shim's one match-data block per pattern is **not** thread-safe; a `prange` kernel needs per-thread state), and packaging (this shim links the system `libpcre2-8`; a wheel would have to vendor it — not a Windows story either).
5. **Per-category masks stay the low-cardinality answer** (10–32 ms, since the codes are already paid for), with §T's cardinality threshold deciding when to fall to lazy or frame. The two are not rivals.

## 8. Reproduce

```
cd experimentation/numba-c-string-matching
./run_all.sh          # builds the shim, runs every benchmark and check, appends to results.jsonl
```

Individual scripts run standalone with `../../.venv/bin/python <script>`. `NUMBA_CACHE_DIR` is not required for the benchmarks; `cache_check.py` sets its own.
