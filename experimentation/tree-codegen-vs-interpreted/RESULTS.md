# Tree codegen vs. one interpreted kernel — measured

**Question:** must a decision tree be compiled by codegen (`decider2.trees.codegen`),
or can one generic `njit` kernel walk an array-encoded tree just as well —
the way decider2's own decision-table engine, and sklearn/XGBoost/GoRules
JDM, already do it?

**Answer: switch trees to the interpreted kernel by default.** At every
size measured, the per-row cost difference is noise against decider2's
20–100 ms flow budget. Codegen's compile cost is not noise: it is
375–435 ms for the *smallest* tree tested and climbs past 7 s at 512
leaves and 20 s at 1,024 leaves — and decider2's own 500-line cap already
refuses trees in this size range: `one_sided_chain_100` (101 leaves) fits
at 416 lines, but `full_binary_d7` (128 leaves) is already over at 524,
so the cap bites somewhere between roughly 100 and 130 leaves for the
shapes measured here. The interpreted kernel pays
one ~14 ms compile for the whole process, ever, and then 0.16–2.3 ms per
new tree shape regardless of size, with `walk_batch.signatures` staying
at length 1 throughout (checked, not assumed). A second finding, found
only by re-running the whole experiment repeatedly rather than trusting
one pass, matters as much as the headline number: **codegen's own
per-row speed becomes unpredictable at exactly the sizes where its
compile time is also becoming unbearable**, while the interpreted
kernel's speed does not move. See "Where they cross over" below — this
is not a hedge, it changes what "crossover" means here.

## Method

Both engines are driven from **one canonical tree object**
(`tree_shapes.py`: `Leaf`/`Num`/`Str`, built once per shape) with two pure
converters:

- `to_schema_tree()` → a real `decider2.trees.schema.Tree` document, fed to
  the **unmodified** `decider2.trees.codegen.emit_tree()` and then compiled
  exactly the way `decider2.compile.driver._try_njit` compiles every step
  it ships: `njit(cache=True)(fn)` then an **explicit-signature**
  `.compile((...))` call (float64 for numeric features/thresholds, int32
  for string features and string-literal params — decider2's own
  dictionary-code convention, doc 05 §1.5 / EXPERIMENTS.md §O). Source
  goes through `decider2.compile.cache.get_or_build()` unmodified, so it
  lands on a real, content-addressed file, exactly as production does.
- `to_flat_tree()` → struct-of-arrays numpy (`kind`/`feat_idx`/`op_code`/
  `thresh`/`pat_start`/`pat_count`/`patterns`/`left`/`right`/`leaf_value`),
  walked by **one** `@njit(cache=True)` kernel (`interpreted_kernel.py`,
  ~70 lines total) — explicit index loop, no recursion, no Python objects,
  no `typed.List`. This one kernel handles all five tree shapes below
  without modification.

Both engines see the **same rows** (same numpy arrays, keyed by feature
name) for a given shape, so `np.array_equal` on the output arrays is a
real check, not a coincidence — asserted for every shape, before any
timing counts. All shapes passed on every run of the whole experiment
(5 runs of the four required shapes plus credit_tree, 5 runs of the
depth-10 supplementary check — see below), 100,000 rows each.

`decider2.compile.cache`'s on-disk artifacts persist across processes by
design — this repo's `experimentation/` directory had none for these exact
trees, but the harness (`run_experiment.py`) also explicitly `rm -rf`s its
build directory at the start of every run, so every "cold" number below is
a genuine first-ever compile, not a replay of a previous run of this same
script.

**Deliberate simplification:** each string feature is tested by at most
one node per tree (documented in `tree_shapes.py`). decider2's codegen
hoists all uses of one string feature across a whole tree into one shared
matcher step; reconstructing that shared step's literal order from outside
`codegen.py`, purely to drive a benchmark row-loop, would add bookkeeping
without changing what's being measured. The credit tree (the only shape
with strings) respects this by construction.

**Not part of decider2's real deliverable:** a small per-tree "row-loop
wrapper" (`codegen_bench.py::_build_row_loop`) that calls the compiled
matcher/path/output functions in a loop over a row matrix. decider2's own
`compile.driver` normally supplies this by fusing a tree into a pipeline's
compiled row loop; standing up a whole pipeline just to drive one tree over
one matrix seemed like the wrong thing to spend the time budget on. This
wrapper's own compile cost is reported **separately**
(`wrapper_build_and_compile_s`, always <4 ms) so it never hides inside the
headline "codegen compile" number, and it embeds threshold/pattern-code
values as generated-source constants purely as harness plumbing — the
tree's own `path_fn` still takes every threshold as a real argument, never
a baked literal, which is the property doc 05 §4.2 actually specifies.

Row count: 100,000 (task guidance: measure one count, don't sweep to 1M).
Compile numbers: wall time via `time.perf_counter()`. ns/row: **minimum**
of 7 repeated calls within one process, not the mean — per the task's
"measure the kernel, not the scheduler" instruction. Memory: `free -h`
checked before every run (never launched below 8 GiB available, this
machine has been OOM-killed before); the largest single process
(depth-10 codegen compile) peaked around 3 GB RSS, transient, nowhere
near the box's 31 GiB.

### Two things found only by re-running the whole experiment, not trusting one pass

**1. A process-unstable row seed (bug, found and fixed).** The original
harness seeded row generation with `hash(shape.name) & 0xFFFF`. Python
randomises `str` hashing per process by default (PEP 456,
`PYTHONHASHSEED` unset) — confirmed directly: three separate
`python -c "print(hash('one_sided_chain_100'))"` invocations returned three
different values. For `full_binary_*` shapes this barely matters (every
root-to-leaf path is exactly `depth` comparisons regardless of which
branch is taken, so the per-row cost is close to data-independent). For
`one_sided_chain_100` it matters a lot: its cost is the *average* number of
nodes walked before a leaf, and that depends entirely on how the sampled
feature values fall relative to the tree's thresholds. Before the fix,
three reruns of the identical script gave codegen ns/row of 33.0, 27.7,
59.5 for this one shape — a 2.15× spread — because each run silently
scored different data. Fixed by seeding with `zlib.crc32(shape.name.encode())`
instead (stable across processes). After the fix, two reruns gave 28.4 and
28.8 ns/row — a 1.4% spread. **This resolved essentially all of
`one_sided_chain_100`'s apparent instability**; it was a row-sampling
artifact, not an engine property. (`run_experiment.py` and
`supplementary_depth10.py` both carry the fix now, with a comment
explaining why.)

**2. Codegen's own ns/row is genuinely unstable at large emitted-code
sizes — the seed fix does *not* explain this one.** `full_binary_d9`
(2,060 lines) and the depth-10 supplementary shape (4,108 lines) both have
data-independent path length by construction, so the seed bug above cannot
be the cause, and re-running with the *same* (now-fixed, reproducible) row
sample still produced very different codegen ns/row: depth-10 gave 241.9,
134.0, 141.3, 241.9 ns/row across four otherwise-identical runs (a >80%
spread) while the interpreted kernel gave 138.4, 140.0, 141.0, 140.1 across
the same four runs (a 1.9% spread) — see "Where they cross over" for the
full table and what this means for the recommendation. Compile *time* at
this size was stable (20.6–21.2 s across the same four runs, ~3% spread) —
it is specifically the compiled function's own execution speed that
becomes unpredictable, not the act of compiling it. This is consistent
with instruction-cache/code-layout sensitivity in a multi-thousand-line
`njit`-compiled function on a shared, noisy machine; not directly
instrumented further here (`perf stat`/`.inspect_llvm()` would confirm the
mechanism, out of scope for this budget).

**Also encountered, and worth naming plainly:** partway through this run,
`git status` showed `decider2/src/decider2/trees/schema.py` as modified
with an uncommitted, in-progress edit (another agent working on this same
branch, exactly as flagged at the start of this task). One depth-10 rerun,
launched right as that file's mtime updated, crashed with
`AttributeError: 'EmitContext' object has no attribute 'column'` —
a transitional, mid-edit state, not a bug in this harness or in the
shipped `codegen.py` (which was untouched throughout: its mtime predates
this entire session). A smoke-test immediately afterward confirmed
`emit_tree()` was working again. Every number reported below comes from a
run whose log timestamp brackets confirm it ran against a stable,
non-mid-edit `schema.py` — checked file-by-file, not assumed — except that
one crashed attempt, which is excluded and not counted among the samples.

Given both of the above, every ns/row number in this report is the
**minimum across 4–5 independent process-level runs** of the full
experiment (not just 7 calls within one process), with the full observed
range quoted alongside it. Where min and the rest of the range tell a
different story (this happens at d9 and d10), both are shown — reporting
only the minimum there would be quietly cherry-picking codegen's best
case, precisely because codegen is the side whose variance is large.

## Results (100,000 rows, cold compile, min of 4–5 full-experiment reruns)

| shape | leaves | nodes | lines | ≤500-line cap? | codegen compile (min–max) | codegen ns/row (min–max) | interpreted ns/row (min–max) | speedup (min-based) |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| credit_tree (mixed numeric+string) | 20 | 39 | 110 | yes | 388–423 ms | 20.7–22.9 | 56.7–61.2 | **2.74×** |
| full_binary_d5 | 32 | 63 | 140 | yes | 375–435 ms | 24.5–26.8 | 64.8–68.7 | **2.65×** |
| full_binary_d7 | 128 | 255 | 524 | **no** | 1,395–1,571 ms | 42.2–49.7 | 85.5–100.7 | **2.03×** |
| full_binary_d9 | 512 | 1,023 | 2,060 | **no** | 7,250–8,123 ms | 82.8–127.0 | 123.0–127.4 | **1.49×** (min); ≈**1.0×** (typical — see below) |
| one_sided_chain_100 (fixed seed) | 101 | 201 | 416 | yes | 1,306–1,367 ms | 28.4–28.8 | 38.2–38.7 | **1.35×** |
| *full_binary_d10 (supplementary)* | 1,024 | 2,047 | 4,108 | **no** | 20.6–21.2 s | 134.0–241.9 | 138.4–141.0 | **1.03×** (min); as low as **0.58×** (worst codegen run vs. interpreted's own worst) |

Raw numbers: `results.jsonl` (last full run's output, one line per
measurement, flushed immediately — a kill loses at most the in-flight
measurement) plus the per-run logs kept for this reproducibility check:
`run.log` (original), `run_rerun1.log`, `run_rerun2.log` (pre-fix reruns,
used to discover the seed bug), `run_fixedseed_1.log`,
`run_fixedseed_2.log` (post-fix, the two runs the headline table's
`one_sided_chain_100` numbers come from). The four depth-10 ns/row samples
(240.9, 134.0, 141.3, 241.9) are in, respectively: `supplementary_depth10.log`
(run 1), `d10_run2_and_main_rerun1_raw.jsonl` (run 2 — a raw `results.jsonl`
snapshot taken before a later main-experiment run would otherwise have
truncated it away, last line), `d10_run3.log` (run 3, pre-fix), and
`d10_fixedseed_1.log` (run 4, the one post-fix sample — `d10_fixedseed_2.log`
is the run that crashed mid-compile, see above, and is excluded from the
range/min but kept for the record). Reproduce with
`<repo>/.venv/bin/python run_experiment.py`; the depth-10 check
separately with `<repo>/.venv/bin/python supplementary_depth10.py`.

**On the compile numbers §P quotes vs. these.** `decider2/docs/EXPERIMENTS.md`
§P *does* exist (a note in an earlier draft of this file said it could not
be found — that was wrong; it was mid-edit by another agent at the time
and has since landed) and gives, for trees built by its own harness during
the decider-1 migration: credit tree 120 lines/0.67 s/104 ns/row;
full-binary d5 84 lines/0.29 s/45.6 ns/row, d7 276/1.02 s/123.8, d9
1044/4.32 s/336.9; one-sided chain-128 278 lines/1.39 s/116.6 ns/row. This
harness's independently-built trees of the same nominal shape are **not**
identical trees (different feature counts, different random
thresholds/structure under the same depth — the two harnesses were never
meant to reproduce each other bit-for-bit) and land at roughly 1.3–2×
*more* emitted lines yet 2–4× *faster* measured ns/row, with compile time
in the same order of magnitude (1.0–1.7× this harness's numbers, either
direction). The direction of the ns/row gap is consistent across every
shape, which suggests a real methodological difference (min-of-N vs.
whatever §P used, or a different row-loop call convention) rather than
noise — but both harnesses agree on the two things that matter for the
recommendation: codegen compile cost is real and grows with tree size, and
codegen's ns/row advantage over a generic kernel narrows as trees grow.
Treat §P's absolute numbers as a sanity check on shape choice and order of
magnitude, not as directly comparable to the table above.

### 1. ns/row at 100k rows

Codegen is reproducibly faster per row for every shape up to and including
depth 7 (524 emitted lines) — 2.0×–2.7×, not a rounding error. Past that
point the picture changes (see §4). In absolute terms even the *widest*
reliable gap (68.7 vs. 24.5 ns/row, full_binary_d5, a 44 ns difference) is
noise against a 20–100 ms flow budget — see "weighed against the budget"
below.

### 2. Compile/build time per tree shape

Codegen: **never free, and stable enough to trust a single number.**
375–435 ms for the two smallest shapes tested (20 leaves, 32 leaves),
climbing past 1.4 s at ~100–130 leaves and to 7.2–8.1 s / 20.6–21.2 s at
512 / 1,024 leaves. Unlike ns/row, compile *time* itself did not show the
large run-to-run swings described above — it is the compiled function's
*execution* speed that gets unpredictable at large sizes, not the act of
producing it. `path_compile_s` and `output_compile_s` are comparable in
size at large leaf counts (depth 9: 3.98 s vs. 3.34 s; depth 10: 10.8 s
vs. 9.9 s) — `output_fn` is its own O(leaves) if/elif chain mapping
`result_idx` to the leaf's value, and at 512–1,024 distinct leaf values
that is a second large function LLVM has to compile, easy to miss if you
only think about "the tree" as the traversal.

Interpreted: **effectively zero, verified, not assumed.** `flatten_s`
(pure-Python tree walk building the arrays) is 160 µs–2.3 ms across every
shape tested, growing with node count but never approaching codegen's
numbers at any size. The kernel itself compiles exactly **once** per
process: `walk_batch.signatures` stays at length 1 across every shape in
every run (checked every time, not assumed once). The one-time compile
was 13–15 ms.

Warm recompile (same tree, on-disk cache populated by a prior run) drops
codegen to 7–39 ms, confirming `decider2.compile.cache`'s own design goal
works as advertised for a tree seen before. It does **not** help a
genuinely new or edited tree, which is what "a new tree config arrives"
actually means, and is the number this experiment was asked for.

### 3. Cold start — "a new tree config arrives" to "it can answer"

This is codegen's real cost, and it is not small even for the tree closest
to what a UI author would actually build: **388–423 ms for a 20-leaf,
mixed numeric/string credit tree** — before a single row is ever scored.
The interpreted kernel's marginal cold-start cost for that same tree is
**~160 µs** (a >2,400× difference at the low end), once its one ~14 ms
kernel compile has been paid — which happens once per process lifetime,
not per tree, and which decider2 could pay unconditionally at process
start the same way it already warms other kernels (doc 02 §3.4).

### 4. Where they cross over

There is a real crossover in ns/row, but re-running the experiment
repeatedly changes what it looks like: it is **not** a clean line the two
curves cross once. It is codegen's own variance opening up.

| depth | codegen ns/row (min–max, n runs) | interpreted ns/row (min–max, n runs) |
|---|---:|---:|
| 5 (n_features×63 nodes) | 24.5–26.8 (n=5) | 64.8–68.7 (n=5) |
| 7 | 42.2–49.7 (n=5) | 85.5–100.7 (n=5) |
| 9 | 82.8–127.0 (n=5) | 123.0–127.4 (n=5) |
| 10 (supplementary) | 134.0–241.9 (n=4) | 138.4–141.0 (n=4) |

Through depth 7 codegen's range and interpreted's range don't overlap —
codegen is unambiguously faster, run after run. At depth 9 the ranges
overlap almost entirely (82.8–127.0 vs. 123.0–127.4): codegen's own
*worst* run (127.0) is slower than its *best* run (82.8) by more than
interpreted's entire range is wide, so whether "codegen wins at d9" is
true depends on which run you happened to catch. Taking the strict
minimum of each side (the brief's own stated methodology), codegen still
wins, 82.8 vs. 123.0 (1.49×) — but the **median** of the five d9 runs is
124.7 (codegen) vs. 125.5 (interpreted): a coin flip, not a win. At depth
10 the two ranges overlap completely and codegen's own worst observed run
(241.9) is slower than interpreted's own worst observed run (141.0) by
1.72×, in the *same* experiment that, on a different run, put codegen
ahead (134.0 vs. 138.4). **Interpreted's range never moves more than ~4%
at any shape tested; codegen's range widens from ~9% at depth 5 to over
80% at depth 10.** That asymmetry, not a single crossover point, is the
finding.

The likely mechanism (not fully instrumented, stated as a hypothesis, not
a proven fact): the interpreted kernel is one small, constant-size
compiled loop regardless of tree shape, so it always fits in instruction
cache and its performance depends only on data-independent array indexing.
Codegen's emitted function grows with the tree — 110 lines at 20 leaves,
4,108 lines at 1,024 leaves — and a multi-thousand-line `njit`-compiled
function is far more exposed to code-layout effects (which basic blocks
land in the same cache line, inlining decisions, branch predictor state
across a much larger working set) that a shared, noisy machine can
perturb run to run. The row-seed bug above shows one *data-dependent*
version of the same class of effect for `one_sided_chain`; this is the
*code-size-dependent* version, and it doesn't go away when the data is
held fixed.

**This crossover is still not the load-bearing finding, for the same
reason as before.** decider2's own 500-line cap refuses `full_binary_d7`
(524 lines) and everything past it *today*, in production, with no
override — so the shapes where codegen's per-row edge is shrinking and
turning unpredictable (d9, d10) are exactly the shapes codegen **cannot
currently compile at all** without raising `LINE_CAP` (which this
experiment did, explicitly, via `emit_tree(..., line_cap=100_000)`, to be
able to measure them). The interpreted kernel has no equivalent limit of
any kind — same ~µs-to-low-ms flatten, same one-time-ever kernel compile,
same tight ns/row range, regardless of shape.

## Weighed against decider2's actual budget

The brief states the budget directly: 20–100 ms per flow, tens of rules.
Measured against that:

- **ns/row is noise everywhere tested, on either side of the crossover.**
  Even the largest reliable per-row gap (44 ns, full_binary_d5) times a
  generous "tens of rules" (say 50 tree evaluations in one flow) is ~2 µs
  — against a 20,000–100,000 µs budget, four to five orders of magnitude
  below the floor. Past depth 9, where codegen's own variance is larger
  than the gap itself, "noise" describes codegen's ns/row *too*, not just
  the difference between engines.
- **Compile time is not noise anywhere tested.** The smallest tree tried
  (20 leaves — smaller than most real credit policies) already costs
  375–435 ms cold, itself already a meaningful fraction of what
  EXPERIMENTS.md §G calls out elsewhere as the budget a live-editing UI
  can tolerate ("ten seconds, up to about 35 rules"). Anything past
  ~130 leaves is refused outright by the current cap; anything the cap is
  raised to admit costs seconds to tens of seconds, with the added property
  (new in this run) that its per-row speed can no longer be predicted
  from one measurement either.

## Recommendation

**Switch decider2's tree engine to the interpreted kernel by default.**
This mirrors the choice the decision-table engine already made, for the
same reason: rows/nodes-as-data, one kernel, compiled once. Concretely:

1. Make `interpreted_kernel.walk_batch`'s pattern (struct-of-arrays,
   `njit(cache=True)`, explicit index loop) the tree engine's real
   implementation, replacing per-shape codegen as the default path.
2. **Do not delete codegen.** Keep it available, opt-in, for a specifically
   measured need this experiment did not find evidence for in decider2's
   stated use case (request-latency-bound flows, tens of rules) but which
   could exist for a genuinely throughput-bound *batch/offline* scoring
   job — millions/billions of rows, no live-editing requirement, where the
   2–2.7× per-row gap at small-to-medium tree sizes turns into real
   aggregate compute cost even though it's invisible at request scale. If
   that use case is ever confirmed, restrict codegen to the size range
   where this experiment actually found it reliably faster
   (≤ full_binary_d7 / ≤524 emitted lines / ≤~130 leaves) — past that,
   this experiment did not find a dependable codegen advantage, only an
   unpredictable one.
3. `LINE_CAP`'s current failure mode (`TreeTooLarge`, doc 05 §7) should
   become the trigger for automatically falling back to the interpreted
   kernel rather than a hard error, if codegen is kept at all — today it
   simply refuses depth-7 full-binary and anything bigger, for a
   limitation that this run's data says is protecting codegen from a size
   regime where it stops being reliably faster, not just from long
   compiles.
4. **New, from this run specifically:** even within the size range where
   codegen keeps a real per-row edge, that edge is a known, bounded
   quantity for the interpreted kernel (its range moves by single-digit
   percent) and an *unknown* one for codegen once trees approach the
   cap (its range moved by 50–80%+ in the shapes where the cap already
   bites). A team choosing "keep codegen for the fast path" should know
   they are also choosing an engine whose speed on a given deploy is not
   fully predictable from a benchmark run at that size.

This is not a hedge — the numbers say switch, plainly, for the case that
matters (decider2's actual flows). The narrow codegen advantage at small
sizes and the batch-job caveat are the honest edges of that conclusion,
not a way of avoiding it.

## What surprised me

- **The crossover isn't a line, it's codegen's own variance opening up.**
  Going in, "is there a crossover" felt like a question with a single
  numeric answer. Re-running the whole experiment 4–5 times (because the
  first depth-10 number didn't reproduce) turned up a different, more
  useful fact: past ~2,000 emitted lines, codegen's ns/row on this machine
  ranges over 50–80%+ between otherwise-identical runs, while the
  interpreted kernel's never moves more than ~4%. "Which engine is
  faster at depth 10" doesn't have a single answer with the codegen path;
  "which engine's speed can you rely on" does.
- **A silent, process-unstable seed nearly hid this.** The harness's
  original `seed=hash(shape.name) & 0xFFFF` looked deterministic and
  reads deterministic — it's only unstable because Python randomises
  string hashing per process by default, which only shows up by actually
  running the same script twice and comparing. It fully explained
  `one_sided_chain_100`'s apparent 2.15× instability (a real, if
  unintentional, lesson in "the workload changed, not the kernel") but
  did *not* explain full_binary_d9/d10's instability, which persisted
  after the fix — worth separating the two, since conflating them would
  have hidden the more interesting (code-size, not data) finding above.
- **`output_fn`'s cost, not just `path_fn`'s.** I expected the path
  traversal function to dominate codegen's compile time; at depth 9 it's
  `path_compile_s` 3.98 s vs. `output_compile_s` 3.34 s — nearly even,
  and at depth 10, 10.8 s vs. 9.9 s. `output_fn` is its own O(leaves)
  if/elif chain mapping `result_idx` to the leaf's value, easy to miss if
  you only think about "the tree" as the traversal.
- **How much the on-disk cache mattered to get an honest number at all.**
  Without the explicit `shutil.rmtree` this harness does at the top of
  every run, a re-run of the same script silently reuses cached
  artifacts and reports a >100× understated "cold" compile. Worth a flag
  for anyone else benchmarking anything in `decider2.compile.cache`'s
  neighbourhood.
- **Running an experiment against actively-edited source is a real
  hazard, not a hypothetical one.** One depth-10 rerun crashed mid-run
  because `decider2/src/decider2/trees/schema.py` was mid-edit by another
  agent on this same branch at that exact moment (confirmed via
  `git status` and file mtimes). It resolved itself within a minute and
  every reported number here is confirmed to have run against a stable
  version of that file — but it's a genuine, disclosed gap: a benchmark
  run against a shared, actively-developed codebase is not automatically
  a benchmark of one fixed implementation.

## Files

- `tree_shapes.py` — canonical tree objects, the four required shapes
  (+ the depth-10 supplementary one), converters to both engines' inputs,
  row generation (now seeded via `zlib.crc32`, process-stable — see the
  Method section's note on the bug this replaced).
- `interpreted_kernel.py` — the one generic kernel (~70 lines).
- `codegen_bench.py` — drives the real `decider2.trees.codegen`/`compile.cache`
  path end to end; the row-loop wrapper is isolated here.
- `run_experiment.py` — orchestrates the four required shapes + credit_tree,
  asserts identical answers, writes `results.jsonl` incrementally, prints
  the summary table. Re-run several times for this report; each run
  truncates and rewrites `results.jsonl` by design (a fresh cold-cache run
  every time), so the per-run `.log` files are what the reproducibility
  claims in this document are checked against.
- `supplementary_depth10.py` — the depth-10 check, made reproducible as a
  script (the ask this task picked up from a previous, rate-limited run of
  it); appends to `results.jsonl` rather than truncating.
- `results.jsonl` — the last full run's measurements, one JSON object per
  line, plus the depth-10 supplementary entry appended.
- `run.log`, `run_rerun1.log`, `run_rerun2.log`, `run_fixedseed_1.log`,
  `run_fixedseed_2.log` — full stdout of five independent runs of
  `run_experiment.py` (the first three pre-date the seed fix; the last two
  are the clean, reproducible baseline the headline table's
  `one_sided_chain_100` row is drawn from).
- `supplementary_depth10.log`, `d10_run3.log`, `d10_fixedseed_1.log`,
  `d10_fixedseed_2.log` — four depth-10 supplementary reruns
  (`d10_fixedseed_2.log` is the crashed run described above, kept for the
  record rather than deleted).
- `d10_run2_and_main_rerun1_raw.jsonl` — a raw `results.jsonl` snapshot
  saved before a later `run_experiment.py` run would otherwise have
  truncated it; its last line is the second depth-10 rerun (134.0 /
  140.0 ns/row), which has no separate `.log` file.
