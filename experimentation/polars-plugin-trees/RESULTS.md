# A decision tree as a polars expression, in Rust — built and measured

**Question (the owner's):** "I'm worried about having everything be float
encoded. It would be nice if we could process the records directly … maybe
even a polars rust extension." So: if a decision tree were a polars
expression implemented in Rust (`pyo3-polars`), would decider2 be a simpler
and more honest thing — and what would it give up? With the owner's later
weighting: **90% of the traffic is single records**, so batch size 1 is the
main case, not a footnote.

**Verdict: it works, it is honest about types, and it is faster than
decider2 at every batch size — but it does not get the single record
anywhere near the 60 µs spec, and the thing that makes it fast is not the
plugin.** Three facts drive that:

1. **The plugin answers `[true, false]` where decider2 answers `[true, true]`.**
   `[2^53, 2^53+1] == 2^53` on an Int64 column: the plugin compares `i64`
   to `i64`; decider2 pushes both through one `float64` array and cannot
   tell them apart. Bools stay bools, strings are matched as strings
   (`==`, prefix, regex) with no dictionary anywhere. That is the whole
   argument for the approach, and it holds (§1).
2. **Single record, end to end (dict in, answer out), one thread:
   236 µs for the plugin vs 580 µs for decider2's `score()`, against a
   60 µs spec.** 2.5× better than decider2 and still 4× over spec — and
   the kernel is ~1 µs of that. The other 235 µs are polars' expression
   engine: ~30 µs of planner per call, ~35–50 µs per plugin call for
   1–4 columns through the FFI, ~20 µs to build the 1-row frame.
   With polars' default 28-thread pool the same call is **~700 µs p50
   with a 270–1800 µs spread**, because polars evaluates a function's
   inputs on the rayon pool even for one row (§2a). This is the number
   the owner needs: a polars expression cannot be the realtime path.
3. **The big-batch win is polars' parallelism, not the plugin's kernel.**
   4-step pipeline, 1M rows: decider2 fused 253 ns/row; the plugin on
   one thread 119; the same expressions under polars' streaming engine
   on 28 threads **17 ns/row**. And a plain `pl.when/then/otherwise`
   rendering of the same trees — no Rust at all — is 59–98 ns/row and
   beats the row-walk plugin on small trees (§2c). The plugin's own
   row-walk loses to polars' vectorised kernels per test (10 ns vs 0.6 ns)
   and only wins back on deep trees, where `when/then` evaluates every
   condition.

**The case against, stated plainly (§5):** owning it costs a 5-minute
cold build, a 252-crate dependency tree, a 5.7 MB wheel per platform and
Python-ABI, `pyo3-polars` pinning an exact Rust `polars` minor, and one
failure mode that **aborts the Python process with no exception** —
a column of a dtype the plugin was not compiled for (Decimal here;
Categorical until I enabled the feature). Against that, the honest-types
property this experiment proves is available *inside* decider2 for the
cost of typed feature arrays (already measured on a branch:
`../SUMMARY-strings-and-types.md`), and the parallel-batch win is
available from pure polars `when/then` for zero Rust. The plugin is the
best engine measured here and I would still not adopt it as decider2's
core: it moves the tree out of numba's kernel and into polars' engine,
whose per-call floor is the same order as decider2's, and it adds a
toolchain decider2 does not have.

Everything below is under `experimentation/polars-plugin-trees/`.
`decider2/src/` was never modified (`git status decider2/src/` is clean).
Every number is in `results.jsonl` (247 records, flushed per line) with
the 1-minute load average recorded alongside the timing-sensitive ones —
this is a shared 28-core box and load ran 10–75 during the night; I
re-ran the batch grid when it dropped to ~10 and say where it did not.

Reproduce: `./run_all.sh` (≈6 min build, ≈20 min benchmarks).

---

## What was built

`decider_trees/` — a Rust crate (`pyo3-polars` 0.28 → `polars` 0.55.2,
`pyo3` 0.29, `regex`, `rayon`; 227 lines of Rust) plus an 88-line Python
package, built with `maturin` into one `abi3` wheel. The tree is data
(kwargs); the features are `Series`. Usage is exactly the brief's:

```python
import decider_trees as dt
tree = [dt.test(0, "<", 5000.0, 1, 2),        # node 0: income < 5000 ? node 1 : node 2
        dt.test(1, ">=", 25, 3, 4),            # node 1: age >= 25   (int64, compared as i64)
        dt.test(2, "prefix", "retail", 3, 4),  # node 2: sector starts with "retail"
        dt.leaf(0, 10.0), dt.leaf(1, 20.0)]    # leaves: index, value
df.with_columns(dt.walk(pl.col("income"), pl.col("age"), pl.col("sector"), tree=tree).alias("path"))
```

- `walk(*cols, tree=)` → Int32 leaf index (null where a tested feature is null)
- `walk_value(*cols, tree=)` → Float64 leaf value
- `walk_value_packed(...)` → same, tree shipped as a 36-byte-per-node blob (§2a)
- `parallel=True` → the row loop is split across polars' own rayon pool
- `to_when_then(tree, cols)` → the same tree as pure polars `when/then`, the oracle
- `noop`, `noop_kwargs`, `noop_packed`, `panic_demo` → probes for §2a / §4

The Python type of the test value picks the dtype the node tests
(`float`→f64, `int`→i64, `bool`, `str`); the Rust side checks the
column's *real* dtype against it and raises — it never casts.

---

## 1. Does it work, and is it honest about types?  Yes.

`q1_types.py`, all asserted before anything was timed.

| test | input dtype | plugin answer | via float64 (what decider2 sees) |
|---|---|---|---|
| `[9007199254740992, 9007199254740993] == 9007199254740992` | Int64 | **`[true, false]`** | `[true, true]` |
| `[True, False, None] == True` | Boolean | `[1, 0, null]` | (decider2: `1.0/0.0`) |
| `sector == "retail"` on `["retail","retail-online","mining","Mining Ltd",None]` | String | `[1,0,0,0,null]` | (decider2: int32 dictionary code) |
| `prefix "retail"` | String | `[1,1,0,0,null]` | not expressible in decider2 |
| `regex (?i)^min` | String | `[0,0,1,1,null]` | not expressible in decider2 |

And the refusals, which are the other half of "honest":

```
node 0 tests 'x' as i64 == 9007199254740992, but the column is f64; cast it explicitly if that is what you mean
node 0 tests 'flag' as bool, but the column is i64
node 0 tests 'sector' as a string (== "retail"), but the column is cat; no dictionary encoding is done for you
node 0 regex "(": regex parse error: … unclosed group
node 0 points at node 1/9, tree has 3 nodes
node 0 tests column #3 but only 1 columns were passed
```

All are `polars.exceptions.ComputeError` naming the node, the column, and
both dtypes; polars appends the expression it happened in (which prints
the `.so` path — ugly but exact). Int32/Float32/Datetime/Enum columns are
refused the same way. **Decimal is not refused; it aborts the process
(§4d).**

---

## 2. Single record first, then parallelism versus fusion

### 2a. Batch size 1 — the main case (µs per call, p50 / min)

`q0_single.py`, `q0_parts.py`, `q0_packed.py`. Same 4-step pipeline as
the batch grid (tree over income/age/sector/verified → `adj = pts1 *
income/1000 + age` → tree over adj/sector → `approve = pts2 >= 15`).
"E2E" is dict in, Python value out — what a service would do.

| path | 1 thread | 28 threads (polars default) |
|---|---:|---:|
| **spec** | 60 | 60 |
| decider2 `score(record)` | 580–610 / 524–562 | 559–600 / 522–558 |
| decider2 `apply(1-row frame)` | 1825 / 1589 | 1877 / 1580 |
| plugin E2E, 4 `with_columns` passes | 306 / 289 | 830 / 378 |
| plugin E2E, the 4 steps as ONE nested expression | 285 / 276 | 852 / 323 |
| plugin E2E, nested + packed tree format | **236 / 217** | 694 / 272 |

Against EXPERIMENTS.md §R: decider2's own `score()` was 352.7 µs for a
trivial pipeline and 1548 µs at 30 steps; this 4-step, two-tree pipeline
lands at 580 µs, in between, as expected. The plugin at 236 µs is under
§R's *trivial* number, and 4× over spec.

**Where the 236 µs go** (1 thread, `q0_parts.py`/`q0_packed.py`):

| part | µs | note |
|---|---:|---|
| `pl.DataFrame([record], schema)` | 19 | dict → 1-row frame |
| `frame.row(0)` / `.item()` | 2 | readback is nothing |
| planner + dispatch floor: `df.with_columns(pl.lit(1))` | 29 | no kernel at all |
| polars' own native kernel `income * 2.0` | 36 | +7 for a real (vectorised) op |
| plugin `noop`, 1 input, no kwargs | 37 | **one plugin call costs the same as a native op** |
| plugin `noop`, 4 inputs | 49.5 | +4 µs per extra Series across the FFI |
| + T1 tree as list-of-dicts kwargs (17 nodes) | 81 (+32) | serde-pickle: ~2 µs/node, *every evaluation* |
| + T1 tree as packed blob | 51.5 (+2) | 36 bytes/node, `from_le_bytes` |
| + compile (dtype checks, regex) + walk 1 row | +1 | **the kernel is ~1 µs** |
| nested expression on an existing 1-row frame (2 trees + arithmetic + threshold) | 189 | ≈ planner + 2 plugin calls + 2 native ops |

Expression construction (`walk_value(...)`, which pickles the kwargs) is
20 µs and is done once, not per call — a service would prebuild it. So
the shape is: **~1 µs of kernel, ~235 µs of polars' per-expression
machinery**, which is the same shape EXPERIMENTS.md §S found for
decider2 ("the engine does not matter for `score()`; marshal and readback
are the entire problem") — only with polars' planner and FFI in the role
of decider2's boundary layer.

**The 28-thread column is the one to read twice.** `noop` with 1 input:
34 µs. With 2 inputs: 82. With 4 inputs: 173 p50, 88 min, p99 in the
milliseconds. polars evaluates a multi-input function's inputs on the
rayon pool regardless of frame length; on a 1-row frame that is pure
dispatch and wake-up latency, and it is noisy on a shared box. A service
scoring single records with this plugin would have to run polars with
`POLARS_MAX_THREADS=1` (or one thread per worker), which is the opposite
of the setting that wins §2b.

**Shipping the tree.** kwargs are pickled once at construction and
deserialised by the plugin on *every* evaluation. As a list of dicts that
is ~2 µs/node: 32 µs for the 17-node T1, 188 µs for a 64-node chain,
**1.34 ms for 512 nodes** — a large tree would cost more to unpickle than
decider2's whole `score()`. The packed format brings 512 nodes to 121 µs
(11×) and T1 to +2 µs. So the cost is not intrinsic to "tree as kwargs",
it is intrinsic to the encoding, and it is the plugin author's problem to
solve, not polars'. The streaming engine does *not* re-ship per morsel
(1M rows, depth-512 chain: in-memory 3064 ms, streaming 3029 ms), so it
is once per call, not once per chunk.

### 2b. Batches of 10 / 100 / 1000 (µs per call, p50; 1 thread | 28 threads)

| rows | decider2 `apply` fused | plugin, 4 passes | plugin, 1 nested expr | plugin nested, streaming engine | pure-polars `when/then`, 4 passes |
|---:|---:|---:|---:|---:|---:|
| 1 | 1825 \| 1877 | 247 \| 696 | 238 \| 698 | 639 \| 752 | — |
| 10 | 1741 \| 1785 | 223 \| 398 | 218 \| 659 | 711 \| 1471 | 1367 \| 1518 |
| 100 | 1790 \| 1826 | 243 \| 430 | 227 \| 667 | 923 \| 1813 | 1303 \| 1467 |
| 1000 | 1913 \| 2099 | 314 \| 479 | 298 \| 864 | 947 \| 1811 | 1324 \| 1578 |

Up to 1000 rows nothing is proportional to rows: decider2's `apply` has a
~1.8 ms floor (its frame boundary — `score()` bypasses most of it), the
plugin's four passes ~220 µs, the streaming engine ~700 µs of its own
setup, and the pure-polars `when/then` form ~1.3 ms because it is ~30
expression nodes, each paying the planner. At these sizes the plugin is
6–8× faster than decider2 `apply()` on one thread — and it is *all*
fixed cost on both sides.

### 2c. Parallelism versus fusion — 10k to 10M rows (ns/row, median of 7; 3 at 10M)

`q2_driver.py`; every cell its own process (polars reads
`POLARS_MAX_THREADS` at import). Answers asserted equal to the
`when/then` oracle in every cell before timing. Load average ≈10 for
10k–1M; it rose to 17–28 during the 10M cells, so treat 10M as ±30%.
(A first pass at load 50–75 is also in `results.jsonl`; it is 2–3× worse
across the board and I do not use it.)

| rows | decider2 fused (1 thr) | plugin eager, 1 thr | plugin eager, 28 thr | plugin lazy in-memory, 28 | plugin **streaming**, 28 thr | streaming, 1 thr | plugin `parallel=True` (rayon inside), 28 | `parallel=True`, 1 thr | pure-polars `when/then`, 28 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k | 464 (4.6 ms) | — | 305 | 274 | 242 | — | 356 | — | 227 |
| 100k | 252 (25 ms) | 128 | 249 | 170 | **37.8** | 131 | 62 | 154 | 59 |
| 1M | 253 (253 ms) | 119 | 126 | 181 | **17.2** | 133 | 34.4 | 302 | 98 |
| 10M | 931 (9.3 s) | 151 | 354 | 185 | **11.0** | 129 | 36.0 | 269 | 63 |

Reading it:

- **Engine difference, parallelism removed** (decider2 vs plugin, both
  one thread): 253 vs 119 ns/row at 1M — the plugin is 2.1× faster.
  Part of that is Rust vs numba (`../rust-tree-interpreter/RESULTS.md`
  measured 12–28% on the walk alone); most of it is that decider2's
  "fused" pipeline is not one loop here: a tree is a *packed* segment
  that never fuses with a neighbour (`compile/driver.py`'s own docstring),
  and each tree also hoists a string-matcher step, so this 4-step
  pipeline runs as five-plus segments with materialised intermediates.
  The `interpreted ≡ stepped ≡ fused` property holds — I checked it
  (§4b) — but fusion is not buying anything on a tree-heavy pipeline.
- **Parallelism**: the same plugin under polars' streaming engine goes
  from 133 ns/row (1 thread) to 17 (28 threads) at 1M — 7.7× on 28
  cores. The in-memory engine does *not* parallelise a single elementwise
  expression (126 vs 119); only the streaming engine's morsels or the
  plugin's own rayon split (34 ns/row) do. So "polars runs them on its
  thread pool" is true only in the streaming engine.
- **The answer changes at 10k rows**: there decider2 (4.6 ms) and every
  polars form (2.3–3.6 ms) are within fixed-cost noise of each other and
  all are inside the 20–100 ms batch budget. From 100k up the streaming
  plugin wins by 7–85× and decider2 is the only engine that leaves the
  budget (253 ms at 1M).
- **decider2 at 10M (931 ns/row) is out of line with its 1M number** and
  was measured at load 18; I would not quote it without a rerun on an
  idle box. The plugin numbers at 10M are consistent with 1M.

**What one row-walk test costs against polars' own kernels** (`q2_isolate.py`,
1M rows, 1 thread, load 15):

| | ns/row |
|---|---:|
| plugin floor (leaf-only tree: kwargs + output) | 9.6 |
| + 1 f64 `<` / 1 i64 `<` / 1 bool / 1 str `==` / 1 str prefix / 1 str regex | +10 / +12 / +14 / +26 / +21 / +45 |
| polars native `income < 5000` / `sector == "mining"` / `str.contains(regex)` / `when(...).then().otherwise()` | 0.6 / 2.4 / 41 / 1.4 |
| T1 (17 nodes) on a 1-chunk frame / on a 10-chunk frame | 66 / 125 |

A scalar row-walk pays ~10 ns per test where polars' vectorised
comparison pays 0.6 — SIMD and no data-dependent branch. The plugin
only walks the *path*, though, and `when/then` evaluates *every*
condition, so which wins depends on tree size (`q2_depth.py`, full
binary trees over 6 f64 features, 1M rows; load rose 18→33 across the
run, so the d9 row is inflated):

| depth (leaves) | plugin row-walk, 1 thr | plugin `parallel=True` | pure `when/then` in-memory | `when/then` streaming, 28 thr | decider2 fused |
|---|---:|---:|---:|---:|---:|
| 3 (8) | 39.5 | 10.8 | **19.6** | 4.5 | 83 |
| 5 (32) | 65.6 | 11.2 | 95.0 | 10.3 | 191 |
| 7 (128) | 206.9 | 46.4 | 867 | 97 | 649 |
| 9 (512) | 214.6 | 70.8 | 2401 | 185 | 1105 |

Crossover is between 8 and 32 leaves. Below it, **pure polars
`when/then` — no Rust, no wheel, no plugin — is the fastest way to run a
tree in polars.** Above it the row-walk wins by up to 11× because
`when/then` is O(nodes) per row. The second surprising line is
decider2's: 649–1105 ns/row at d7/d9 against the standalone numba walker's
85–120 in `../rust-tree-interpreter/RESULTS.md` — the packed-segment
plumbing around the walk, not the walk, is what decider2 pays (and this
was measured under load; the ratio to the plugin is the reliable part).

The 10-chunk line matters too: a plugin receives whatever chunk layout
the engine hands it. Mine rechunks (one copy) — on a frame the in-memory
engine has chunked by earlier parallel work, that copy doubled the cost.
A production plugin would walk chunk-by-chunk, more code.

---

## 3. What it costs to own — measured

| | this plugin (`pyo3-polars`) | plain PyO3 crate (`../rust-tree-interpreter`, §3) |
|---|---:|---:|
| first build, crates not downloaded | **5 min 38 s** wall, 47 CPU-min, 869% CPU, 2.24 GB peak RSS (dependencies only — my own 40 lines failed to compile at the end; +13 s for those) | 29 s |
| cold build after `cargo clean`, crates cached | **4 min 39 s** (278.6 s), 48 CPU-min, 1074% CPU, 2.6 GB RSS | 14.7 s |
| turning on one dtype feature (`dtype-categorical`) | **4 min 38 s** — 11 crates (polars-core and everything above it) recompile | n/a |
| incremental rebuild, one source line | 5.7–10.2 s | 0.87–1.5 s |
| wheel, release | 6.73 MB unstripped / **5.71 MB stripped** (the `.so` inside: 29.2 / 21.8 MB) | 888 KB stripped |
| platform tag | `cp310-abi3-manylinux_2_34_x86_64` (`abi3-py310` on, so one wheel per platform, not per Python minor) | `cp314-cp314-manylinux_2_34_x86_64` |
| dependency count | **252 unique crates** (718 edges), 37 of them `polars-*`; 4 versions of `syn`, 3 of `hashbrown`, 2 of `tokio` | 50 |
| `target/` on disk | 1.4 GB | — |

The wheel is 6.4× the size and 8× the crate count of the plain PyO3
extension because **a plugin statically links its own copy of polars**
— `polars-core`, `polars-arrow`, `polars-plan`, plus, once
`dtype-categorical` is on, `polars-lazy` and through it `tokio`. The
Python `polars` you `pip install` is a second, separate copy; the two
talk only through the Arrow C data interface.

**When polars is upgraded.** Two different questions:

- *Python-side upgrade* (`pip install -U polars`): the plugin keeps
  working as long as the FFI protocol matches. The `.so` exports
  `_polars_plugin_get_version` (here 0.1); `polars-plan`'s loader
  (`src/plans/aexpr/function_expr/plugin.rs`) accepts major 0 and
  dispatches on the minor, else `ComputeError: this Polars engine doesn't
  support plugin version`. So an old plugin against a newer polars is a
  clean error at worst, and the version pair in this experiment —
  Python polars **1.41.2** loading a plugin built on Rust polars
  **0.55.2** — is already such a mismatch, and works. I did not test an
  actual upgrade across an FFI bump; polars' own guide says it is rare.
- *Rust-side upgrade* (`pyo3-polars` bump): `pyo3-polars` 0.28 pins
  `polars = ^0.55.1` and `pyo3 = 0.29` — the plugin's polars version is
  chosen by `pyo3-polars`, not by you. Each bump is a 5-minute full
  rebuild plus whatever API moved: in 227 lines I hit two moves already
  (`arrow` is not re-exported via `polars::prelude` with
  `default-features = false`; the rayon pool is now
  `polars_core::runtime::THREAD_POOL`, not `POOL`). A stale local crates
  index also produced a spurious "no matching version of polars-lazy
  0.55.2" until `cargo update` refreshed it — 10 minutes lost to
  tooling, the kind of thing this repo's numba-only build never has.

This bears on the owner's stated preference for "depend on a maintained
library and pull the latest": the *plugin API* is maintained and stable
(one macro, one FFI version number, a clean error on mismatch), and that
is genuinely good. But the *plugin itself* is still ~300 lines the owner
writes and owns, with a Rust toolchain, a wheel matrix and a 5-minute
rebuild attached — `pyo3-polars` maintains the bridge, not the tree.

---

## 4. What breaks or gets harder

**(a) The single-record path** — §2a. 236 µs at best on one thread,
~700 µs and noisy on polars' default thread pool; 4× over spec; 99% of
it is polars' per-expression machinery, ~1 µs is the tree.

**(b) `interpreted ≡ stepped ≡ fused`.** The equivalent property is
"plugin ≡ pure-polars `when/then` ≡ plugin under any engine/thread
count", and it is checkable the same way (`q4_breaks.py`): on 3 seeds ×
50k rows with nulls sprinkled into every feature, plugin eager == the
`when/then` oracle == plugin streaming+`parallel=True`, and decider2's
own interpreted == stepped == fused == plugin on 50k all-present rows.
One caveat that is not the plugin's: `adj = pts1*income/1000 + age`
differs by 1 ulp between numba and polars on 8.8% of rows (max relative
2.2e-16, `q4_numeric_divergence`) — the compilers round differently.
Every tree output and the final `approve` were bit-identical.

**(c) Retuning without recompiling.** Nothing compiles: a retune is a new
kwargs blob (`count_new_compiles()` = 0 across three retunes, trivially;
decider2 also 0 across three retunes — its one compile event fired on the
first `apply()` after `precompile()` on a differently-sized frame, not on
a retune). What retuning *costs* is the per-call deserialisation in §2a:
+2 µs/node as dicts, +0.2 µs/node packed. A tree edited by a UI would
need to be re-pickled into the expression (20 µs) — no more.

**(d) Errors, panics, aborts.** Three outcomes, demonstrated in
subprocesses (`q4b_retune_panic.py`, `panic_worker.py`):

| what | outcome |
|---|---|
| bad tree / dtype mismatch / bad regex | `ComputeError` with the messages in §1; the *next* call in the same process works |
| `panic!` inside the plugin body (`panic_demo`: index out of bounds) | **caught** by `pyo3-polars`' `catch_unwind`: `ComputeError: the plugin panicked` — the panic's location and message are hidden unless `POLARS_VERBOSE=1` |
| a column of a dtype the plugin's polars was not compiled with (Decimal; Categorical before I enabled `dtype-categorical`) | **process abort, SIGABRT, no Python exception** — polars-core panics while importing the Series over the FFI, *before* the macro's `catch_unwind`, and a panic in that `extern "C"` frame cannot unwind. stderr shows only `thread caused non-unwinding panic. aborting.`; the cause (`Arrow datatype Dictionary(UInt32, Utf8View, false) not supported by Polars. You probably need to activate that data-type feature`) appears only with `POLARS_VERBOSE=1` |

The third one is the one to weigh. It is the same class of failure as
`../rust-string-matching`'s unprotected-panic demo, but it is not under
the plugin author's control — it is in the FFI import path of the copy
of polars linked into the plugin — and the trigger is an ordinary
upstream dtype arriving in a frame. Debuggability otherwise: `RUST_BACKTRACE`
works, error messages are good, but there is no Python-level stack into
the tree walk and no numba-style `inspect_types`; you debug in Rust.

---

## 5. The verdict, with the case against

**Does decider2 become simpler and more honest if trees are polars
expressions?** More honest, yes — provably (§1). Simpler, no.

*What it would be.* A tree is a `Series`-in, `Series`-out function that
sees real dtypes, with no boundary layer, no float64 gather, no
dictionary encoding, no packed segments; the pipeline is a polars
expression a reader can print; parallelism is free from polars'
streaming engine; the failure surface is one macro and one FFI version
number that a maintained project owns. The Rust is 227 lines and the
kernel inside it is 20. For batch work it is the fastest thing measured
in this whole series (17 ns/row at 1M on 28 threads).

*What it would give up.* Everything that currently makes decider2
decider2 lives in the numba boundary: `interpreted ≡ stepped ≡ fused`,
`precompile()`, `count_new_compiles() == 0`, `on_missing_input`
policies, `param()` schemas, the `<name>_path` column, `explain()`. A
plugin has none of those; the *equivalents* exist (§4b–c) but would be
new code. And the realtime path — the owner's 90% — does not get fixed:
236 µs is better than 580 µs and still not 60 µs, because it is the same
shape of problem (per-call machinery around a 1 µs kernel) in a
different engine, plus a thread pool that has to be turned *off* for
single records and *on* for batches.

*The case against, in one paragraph.* You would add a Rust toolchain, a
5-minute build, 252 crates, a 5.7 MB wheel per platform, an exact
`polars` pin chosen by `pyo3-polars`, and a process-abort on an
unexpected dtype — to get two things: honest types, which typed feature
arrays already deliver inside decider2 for ~20% of walk cost
(`../SUMMARY-strings-and-types.md`), and parallel batch speed, which
plain `pl.when/then/otherwise` delivers for small trees with zero Rust
and which no realistic batch here needs (decider2 is inside its batch
budget at 10k–100k rows; it leaves it at 1M). What the plugin uniquely
wins — deep trees at scale (§2c, 11× over `when/then` at 512 leaves) —
is not a stated requirement.

*If the owner wants polars anyway.* The honest sequence is: (1) write
trees as pure polars `when/then` first — typed, parallel, no toolchain —
and measure whether tree size ever crosses the ~32-leaf line; (2) only
then reach for this plugin, with the packed wire format, walking
chunk-by-chunk, built with every dtype feature on (`dtype-full`) so the
abort in §4d becomes an error, and served with `POLARS_MAX_THREADS=1`
per worker for single records. And keep `score()` off polars entirely:
neither engine reaches 60 µs through a DataFrame.

---

## What the code looks like

The whole plugin is `decider_trees/src/lib.rs` (227 lines) and
`decider_trees/decider_trees/__init__.py` (88 lines). The parts that
matter, verbatim:

**The wire node and the typed node** — the tree arrives as pickled
kwargs, one flat list; `serde` deserialises it into this:

```rust
#[derive(Deserialize, Debug)]
struct WireNode {
    leaf: Option<i32>, value: Option<f64>,
    col: Option<usize>, op: Option<String>,
    f: Option<f64>, i: Option<i64>, b: Option<bool>, s: Option<String>,
    then: Option<usize>,
    #[serde(rename = "else")] otherwise: Option<usize>,
}
#[derive(Deserialize)]
struct TreeKwargs { nodes: Vec<WireNode>, #[serde(default)] parallel: bool }

enum Test { F64(usize, Cmp, f64), I64(usize, Cmp, i64), Bool(usize, bool),
            StrEq(usize, String), StrPrefix(usize, String), StrRegex(usize, Regex) }
enum Node { Test { test: Test, then: usize, otherwise: usize }, Leaf { idx: i32, value: f64 } }
```

**Where "honest about types" lives** — `compile()` checks each node
against the real dtype of the Series it names, and refuses rather than
casts (one arm shown):

```rust
(None, Some(v), None, None) => {
    polars_ensure!(matches!(dt, DataType::Int64), SchemaMismatch:
        "decider_trees: node {k} tests '{name}' as i64 {op} {v}, but the column is {dt}; \
         cast it explicitly if that is what you mean");
    Test::I64(col, Cmp::parse(op)?, v)
}
```

**The kernel** — a typed view per column, one loop, no allocation; a null
at a tested node makes the row's answer null:

```rust
enum Col<'a> { F64(&'a [f64], Option<&'a Bitmap>), I64(&'a [i64], Option<&'a Bitmap>),
               Bool(&'a BooleanArray), Str(&'a Utf8ViewArray) }

#[inline(always)]
fn eval_test(t: &Test, cols: &[Col], i: usize) -> Option<bool> {
    match t {
        Test::F64(c, op, v) => match cols[*c] { Col::F64(xs, va) => valid(va, i).then(|| op.eval(xs[i], *v)), _ => unreachable!() },
        Test::I64(c, op, v) => match cols[*c] { Col::I64(xs, va) => valid(va, i).then(|| op.eval(xs[i], *v)), _ => unreachable!() },
        Test::Bool(c, want) => match cols[*c] { Col::Bool(a) => (!a.is_null(i)).then(|| a.value(i) == *want), _ => unreachable!() },
        Test::StrEq(c, v)   => match cols[*c] { Col::Str(a) => (!a.is_null(i)).then(|| a.value(i) == v.as_str()), _ => unreachable!() },
        Test::StrPrefix(c, v) => match cols[*c] { Col::Str(a) => (!a.is_null(i)).then(|| a.value(i).starts_with(v.as_str())), _ => unreachable!() },
        Test::StrRegex(c, re) => match cols[*c] { Col::Str(a) => (!a.is_null(i)).then(|| re.is_match(a.value(i))), _ => unreachable!() },
    }
}

#[inline(always)]
fn walk_row(nodes: &[Node], cols: &[Col], i: usize) -> Option<(i32, f64)> {
    let mut n = 0usize;
    loop {
        match &nodes[n] {
            Node::Leaf { idx, value } => return Some((*idx, *value)),
            Node::Test { test, then, otherwise } => { n = if eval_test(test, cols, i)? { *then } else { *otherwise }; }
        }
    }
}
```

**The polars expression** — this is all `pyo3-polars` asks for:

```rust
#[polars_expr(output_type=Int32)]
fn walk(inputs: &[Series], kwargs: TreeKwargs) -> PolarsResult<Series> {
    let v = run(inputs, &kwargs, |r| r.map(|(i, _)| i))?;
    Ok(Int32Chunked::from_iter_options(PlSmallStr::from_static("leaf"), v.into_iter()).into_series())
}
```

where `run` compiles the tree, rechunks each input to one chunk, builds
the `Col` views, and runs `walk_row` over `0..n` — serially, or through
`polars_core::runtime::THREAD_POOL.install(|| (0..n).into_par_iter()...)`
when `parallel=True`.

**The Python side** — the tree is a list of dicts, the call is one
function:

```python
def walk(*cols, tree: list[dict], parallel: bool = False) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB, function_name="walk", args=list(cols),
        kwargs={"nodes": tree, "parallel": parallel}, is_elementwise=True,
    )
```

That is the entire API surface: one `#[polars_expr]` per output dtype,
one `register_plugin_function` per Python function, and a
`#[pymodule]` that exports nothing but `__version__`.

---

## Files

| | |
|---|---|
| `decider_trees/` | the crate: `Cargo.toml`, `src/lib.rs`, `pyproject.toml`, `decider_trees/__init__.py` (+ built `_internal.abi3.so`), `dist/` wheel, `dist_stripped/` |
| `trees.py` | the 4-step pipeline written once as specs, rendered as plugin exprs, pure-polars `when/then`, and decider2 `Tree` documents + steps |
| `q1_types.py` | §1 |
| `q0_single.py`, `q0_parts.py`, `q0_packed.py` | §2a–b: single record and small batches, decomposed |
| `q2_driver.py` / `q2_bench.py`, `q2_profile.py`, `q2_isolate.py`, `q2_depth.py` | §2c |
| `q3_cost.sh` (+ `q3_cost.log`, `build_*.log`) | §3 |
| `q4_breaks.py`, `q4b_retune_panic.py`, `panic_worker.py` | §4 |
| `results.jsonl` | every measurement, with timestamps and load average |
| `run_all.sh` | reproduce everything |
