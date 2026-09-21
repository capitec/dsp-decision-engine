# GoRules ZEN / JDM — could decider2 use it, or learn from it?

Ground rules for this note: everything under it is runnable
(`experimentation/zen-jdm-integration/`), every identical-answers claim is an
`assert` that ran, and every "does not map" claim was produced by trying to
build the thing and reading the actual failure, not by reading the docs and
guessing. `decider2/src/` was read, never modified. `zen-engine==2.0.2`
(published 2026-08-22, Python 3.14, `pip install` via
`VIRTUAL_ENV=.venv uv pip install zen-engine` — installed cleanly, no native
build step, ~10 MB wheel).

Desk research (node types, expression language, hit policies, governance
surface) is already covered in `decider2/docs/research/decision-engine-landscape.md`
§8 and is not repeated here except where a measurement corrected or sharpened
it.

---

## Q1 — Could ZEN be the execution engine, and what would it cost?

**Setup** (`q1_benchmark/`): the identical rule set, defined once
(`rules.py`, 12 contiguous income bands -> tier code + limit) and translated
mechanically into a decider2 `DecisionTable` (`build_decider2.py`,
`BetweenExpression`) and a JDM `decisionTableNode` (`build_zen.py`, targeted
range unary tests — the same shape as ZEN's own `test-data/table.json` and
`credit-analysis.json`'s "Turnover" node). A third, independent
implementation (`rules.oracle()`) is the tie-breaker so both engines being
wrong in the same way can't pass as "identical." `bench.py` runs all three.

**Identical answers — asserted, not assumed.** 2,000 single-record calls and
a 100,000-row batch, boundary-heavy (10% of samples land exactly on a band
edge): decider2, ZEN, and the independent oracle agree on every value,
`tier_code` and `limit`, at every boundary (`0`, `4999`, `5000`, `149999`,
`150000`, …). `q1_benchmark/results.jsonl`, `phase: identical_answers` /
`identical_answers_batch`, both `PASS`.

### Batch, 100,000 rows

| engine | method | ns/row | rows/s | vs decider2 |
|---|---|---:|---:|---:|
| decider2 | `pipeline.apply(mode='fused')` — full boundary, polars in/out | 6,166 | 162,168 | 1.0× |
| ZEN | `ZenEngine.evaluate_batch()`, static loader, one request/row | 9,585 | 104,329 | **1.55×** |
| ZEN | per-row Python loop, `ZenDecision.evaluate()` | 1,394,180 | 717 | **226×** |

Both ZEN numbers are single-threaded — confirmed by rerunning
`evaluate_batch` with `RAYON_NUM_THREADS=1` (8,765 ns/row, statistically the
same as the 28-core default run), so the ~145× gap between ZEN's two batch
shapes is not multicore parallelism, it is **Python-level per-call
dispatch**: `evaluate_batch` crosses the Python/Rust boundary once for the
whole batch and loops inside Rust; the naive loop pays PyO3 argument
marshalling and a GIL acquire/release 100,000 times.

**This corrects an assumption, not just decider2's numbers.** EXPERIMENTS.md
§B's reference point — `objmode` per row at 77× the all-njit baseline — was
this task's expectation for "a per-row FFI call into Rust." The naive ZEN
loop is **226× decider2's real, boundary-included `apply()`** (worse than
§B's number, and §B's baseline was the idealized bare kernel, not a real
`apply()` call, so the true multiple against a from-scratch all-njit-serial
baseline is larger still). But ZEN's own best batch API is **1.55×**, not
226×, or 226/1.55 ≈ 145× better than the naive shape the research note (and
this task's own framing) treated as *the* ZEN-in-batch number. **The FFI
call itself is not the expensive part — the per-call Python binding
overhead is, and ZEN's own API has a way around it that the docs cited in
§8.4 do not mention** (`evaluate_batch` is absent from
`docs.gorules.io/developers/overview/performance.md`; found by reading the
`.pyi` stub, `.venv/lib/python3.14/site-packages/zen/__init__.pyi`, not the
docs site).

### Single-record latency (the realtime path, 20–100 ms budget)

| engine | method | p50 | p95 | p99 | % of 20 ms (p50) |
|---|---|---:|---:|---:|---:|
| decider2 | `pipeline.score()` | 352.7 µs | 434.9 µs | 878.3 µs | 1.76% |
| ZEN | `ZenDecision.evaluate()` (sync) | 1,264.3 µs | 2,097.4 µs | 2,572.9 µs | 6.32% |
| ZEN | `ZenDecision.async_evaluate()`, persistent event loop | 94.6 µs | 331.0 µs | 486.8 µs | 0.47% |

**A second correction, found while chasing why the sync number was so much
worse than the batch-API number for the identical rule.** Testing a
completely empty `passthrough.json` graph (`input -> output`, zero nodes to
evaluate) against the sync `evaluate()` gave **1,218 µs p50 — statistically
the same as the full 12-row decision table (1,264 µs)**. The cost is fixed
per call and independent of the graph; it is not rule evaluation, it is
binding overhead. Switching to `async_evaluate()` inside one reused asyncio
event loop (no other change) dropped it to 71.3 µs on the same passthrough
graph and 94.6 µs on the real rule — a **13–17× reduction**, strongly
consistent with the sync wrapper paying a fresh async-runtime bootstrap
(or equivalent) on every call that the async path, run inside a loop
that already exists, does not. `q1_benchmark/results.jsonl` has all four
single-record rows; the passthrough isolation was a standalone probe (not
re-run into the harness, reported here directly:
p50 1218.4 µs sync / 71.3 µs async on `fixtures/passthrough.json`).

This is the single most decision-relevant number in this section: **decider2's
own current, unoptimized `pipeline.score()` (352.7 µs) beats ZEN's obvious,
documented Python entry point (`evaluate()`, 1,264.3 µs) by 3.6×, but loses
to ZEN's async entry point (94.6 µs) by 3.7×** — and the "obvious" entry
point is exactly what a synchronous Python caller reaches for first. A team
adopting ZEN who does not know to reach for `async_evaluate()` inside a
persistent loop would report ZEN as slower than it is by more than an order
of magnitude, entirely from an API choice.

### Cold start: config arrives -> first answer

| engine | doc build | compile/parse | first call | **total** |
|---|---:|---:|---:|---:|
| decider2 (`table_module`, cold cache) | 0.20 ms | 3.2 ms | 1,011.0 ms | **1,014.4 ms** |
| ZEN (`create_decision`) | 0.10 ms | 0.29 ms | 4.4 ms | **4.8 ms** |

decider2 pays ~1 s to JIT-compile before the first answer (numba compiles
lazily, on first call); ZEN answers in under 5 ms, ~210× faster cold. This
is the expected shape and decider2's own docs already price it in — compile
happens at `decider2 build` (image build time, doc 02 §3.4), never at
request time, so in decider2's actual deployment model this cost is paid
once, offline, not per-deploy or per-request. It matters directly for one
real scenario the research note also names (§9.9 / the landscape note's "a
rule UI can promise ten seconds" work, `EXPERIMENTS.md` §G): an author
editing a rule and wanting to see the new answer *now*, in a browser, before
committing to a build. ZEN's near-zero parse-and-run cost is a genuine
advantage there that no amount of decider2 build-time engineering removes,
because the two are solving different problems (compile-once-run-forever
vs. edit-and-see-immediately).

### Verdict on Q1

**No, not as the execution engine for decider2's compiled kernel — but the
margin is much narrower and more API-shaped than "per-row FFI is fatal"
suggested going in.** At the batch/throughput job decider2 exists for, ZEN's
*best* API is 1.55× slower, not the 145–226× a naive per-row call would
suggest; decider2 still wins, but not by an order of magnitude, and not for
the reason expected (the FFI call itself is cheap; it's specifically the
Python binding's per-call setup that's expensive, and ZEN's own batch API
mostly avoids it). At the realtime single-record job, which the framework's
own docs (doc 01 §6.1) call *primary*, the answer inverts depending on which
of ZEN's two Python entry points is used — a finding that did not exist
before this measurement and is not in the research note or the ZEN docs
consulted for it.

---

## Q2 — Could JDM be an interchange format?

Two converters (`q2_converter/`), each built to the point where it was
tried against a real document and read where it broke, not designed from
the spec alone.

### JDM `decisionTableNode` -> decider2 `DecisionTable` (`jdm_to_decider2.py`)

Run against `fixtures/table.json` (the ZEN repo's own minimal example) and
every `decisionTableNode` in `fixtures/credit-analysis.json` (the ZEN repo's
own credit-risk sample — real content, not written for this task).

| JDM table | result | why |
|---|---|---|
| `table.json` "Hello" | **converts**, verified identical to ZEN on 4 records | single numeric column, trailing wildcard row -> `default` |
| credit-analysis "Company Type" | **converts**, verified identical to ZEN on 4 records | string-set column (isin), trailing wildcard row -> `default` |
| credit-analysis "Turnover" | **refused**, precisely | row `"[200_000..1_000_000]"` is closed on *both* ends; decider2's `BoundMode` is one global choice (`lower_inclusive` XOR `upper_inclusive`) for the whole table and can never make one row closed-both while another (`"> 1_000_000"`) is open. **Demonstrated live**, not just asserted: evaluating the real JDM document at `turnover = 1,000,000` exactly returns `"amber"`; the naive `lower_inclusive` conversion the checker refused to emit would have returned `"green"` at that exact value (`run_table_conversions.py`'s `demonstrate_turnover_boundary`). A silent converter would have shipped a one-value boundary bug. |
| credit-analysis "Country" | **refused**, precisely | the `Company Country` column is wildcarded on a *non-trailing* row (the "amber" row tests `isEu` and leaves country blank). decider2's `InExpression` has no per-row skip — an empty per-row value set means "never matches," the opposite of JDM's blank-cell wildcard (`EqExpression` *does* have a per-row skip, via a `None` value; this table needed it on an In-shaped column, which doesn't have one) |
| credit-analysis "Overall" | **refused**, precisely | the same pattern on four range-shaped columns (`Red`/`Amber`/`Green`/`Critical`), each tested on a different row and blank on the others — a genuinely sparse per-row test matrix, which `BetweenExpression`'s full-row-coverage requirement cannot represent at all |

**What did not map, restated as the general findings** (found by building the
checker, not assumed): (1) decider2's `BetweenExpression` applies one bound-
inclusivity choice to the *entire table*; JDM lets every cell pick its own
bracket. (2) decider2's row *position* doubles as band *identity*
(ascending order is required, contiguity is checked) — JDM's `hitPolicy:
"first"` rows are in business-priority order, which for the Turnover table
is not ascending at all (`> 1_000_000` is listed *first*). The converter
has to re-derive ascending order from each row's own stated bounds before
it can even attempt the inclusivity check. (3) Only `EqExpression` has a
genuine per-row wildcard (`None` -> skip); `InExpression` does not, so any
JDM column that is wildcarded on a non-trailing row *and* needs more than
one value on some other row has no decider2 form. (4) A JDM output cell
that is a computed expression (`input.amount * 0.1`) has no decider2
equivalent — `outputs` names literal columns, never formulas. (5)
`hitPolicy: "collect"` (and per-column `[]` collect) has no decider2
equivalent at all — decider2 tables are first-match-wins, full stop.

**What decider2's table model has that a straight JDM cell does not**: the
`InExpression`/`EqExpression` split, though it cost the "Country" table
above, is also decider2 correctly distinguishing two different runtime
representations (a CSR membership scan vs. a single compare) that JDM's
uniform "cell is a unary test" text does not distinguish at the schema
level at all — that distinction is exactly what lets a numba kernel avoid a
string-set scan when a plain equality will do.

### decider2 `Tree` -> JDM (`decider2_tree_to_jdm.py`)

A tree maps to a *chain of `switchNode`s*, not to `decisionTableNode` — the
better structural match, because neither side forces its row/column shape
onto the other. Built a representative tree (`example_tree.py`: a
`CompositeNode` OR over two different features, a 3-band `CasesRanges`, a
`UnaryNode` string match, and leaves that share an output row via
`result_idx`) end to end: converted it, loaded the JDM document into a real
`ZenEngine`, compiled the *same* tree through decider2's real `tree_module`,
and checked both against a third independent oracle across 72 input
combinations. **All 72 agree**, across all three
(`run_tree_conversion.py`).

Two bugs surfaced only by actually running this, both fixed and now
documented as findings rather than left as assumptions:

- **Field references inside a JDM graph are NOT prefixed `input.`.** The
  object flowing between nodes is the flat upstream object itself; `input`
  is the *display name* the editor gives the entry node, not a wrapper key
  in the data. (Confirmed independently against `credit-analysis.json`'s
  own `"field": "company.turnover"`, never `"field": "input.company.turnover"`.)
  A first attempt using `input.<field>` inside `switchNode` conditions
  silently fell through to every node's default branch — no error, just
  wrong answers, on every single row. This is exactly the kind of
  "silently produces a plausible-looking wrong answer" failure mode this
  task's identical-answers discipline exists to catch.
- **decider2 trees are single-parent trees, not DAGs.** Pointing two
  different edges at the same node (attempting to reuse a "declined" leaf
  from two branches) raises `ValueError: tree ... revisits node ...` —
  decider2's line-cap accounting (doc 05 §7) depends on every node being
  emitted exactly once. JDM's graph has no such restriction — an edge can
  target any node, which is exactly how the tree->JDM converter reproduces
  decider2's leaf-value dedup (several leaves sharing one `result_idx`) as
  several edges converging on one shared `expressionNode`, something the
  *decider2 side of the same tree* is not allowed to do structurally.

**What did not map, tree direction**: `InputRef`-valued thresholds
(decider2's named, hot-swappable, separately-versioned params — doc 02 §4,
doc 08 §4) have to be resolved to literals; the converter raises rather than
guess, and the *name* and the "retune without recompiling" property are
both gone in the JDM copy — changing a threshold there means editing node
content and republishing the document, not swapping a params dict.
`CasesStringMatch`/`UnaryStringMatch` with anything other than exact,
case-sensitive matching is refused outright (ZEL's `matches()`/
`fuzzyMatch()` were not probed against decider2's semantics — refusing to
guess rather than emitting silently-wrong ZEL). `TreeOutput.default` (the
`-1` sentinel for "nothing matched") has no receiving node on the JDM side,
because a decider2 tree can never actually reach it through normal
traversal (every declared branch has a required target) — noted, not
silently dropped.

**What JDM has that a decider2 tree cannot express, stated evenhandedly**:
JDM's graph is a genuine DAG — convergent paths, node reuse, shared
sub-decisions addressed by id — where decider2's tree is deliberately a
tree (doc 05 §7's line-cap accounting is *why*: a shared subtree would
duplicate its emitted lines). This is JDM being structurally richer at the
graph level, for a real reason (governance/authoring flexibility), traded
against decider2 getting a hard, checkable bound on emitted code size that
a DAG-shaped format cannot offer the same way.

### Q2 verdict

JDM is a genuinely serviceable **read** target for the subset that maps —
both fixtures that converted did so with zero manual correction, and the
converter's refusals are exact enough to hand to a human as a work order
("this row states both bounds inclusive, which BoundMode can't do"). It is
a **poor write target for decider2 trees' one governance feature that
matters most** — the params/values split (doc 02 §4) collapses to nothing
in JDM, which bakes every threshold into document text. Treating JDM as
*the* interchange format would mean decider2 either gives up hot-swappable
params for anything exported to JDM, or invents its own non-standard
extension to JDM's content field to carry them — at which point it is no
longer speaking the standard, only its own dialect of it.

---

## Q3 — What should decider2 steal from ZEN's design, regardless?

Ranked by how directly it transfers to decider2's actual constraints
(numba-compiled kernel, byte-identical codegen, params/values split,
always-on taps).

### Steal

1. **Per-column bitset indexing for large decision tables.**
   `core/engine/src/nodes/decision_table/index.rs` — `TableIndex`/
   `ColumnIndex`, one `FixedBitSet` per distinct literal value per column
   (string/number/bool), built once when a table has ≥ `MIN_INDEX_ROWS`
   (8) rows, consulted to narrow candidate rows before evaluating any
   cell. decider2's table kernel currently does a full per-row scan
   unconditionally (`tables/codegen.py`); nothing about that scan is wrong
   for the row counts decider2 has measured so far (EXPERIMENTS.md §P:
   table compile cost is independent of row count), but a bitset index is
   exactly the sort of thing that could live in `shared` — built once at
   `table_module()` time, no kernel recompile, no cost to the "editing
   rows is free" property doc 08 §3.4 already guarantees — and would only
   help the row counts that would ever need it (ZEN's own 8-row floor is a
   reasonable transplant of "don't bother below this size," worth
   confirming against decider2's own numbers rather than copying the
   constant blind). Concrete, bounded, and doesn't touch the kernel's
   determinism story.
2. **A `collect` hit policy.** decider2 tables are first-match-wins only;
   `docs.gorules.io/learn/authoring/decision-tables.md`'s `collect` (every
   matching row, or a per-output-column collect via a `[]`-suffixed field)
   is a real semantic decider2 doesn't have and a real credit/fraud use
   case does ("which of these N flags apply" — not "the first one").
   This is a feature gap, not a performance idea, and it composes cleanly
   with decider2's existing `Expression`/DNF machinery — the hard part
   (row-scan, condition evaluation) is unchanged; only the "stop at first
   `ok`" becomes "collect every row where `ok`."
3. **`evaluate_batch`'s shape as the argument for how NOT to build a batch
   API, if decider2 ever exposes one across an FFI boundary of its own.**
   Not a ZEN-specific idea to copy so much as a confirmed general
   principle, found by measuring: a 145× gap between "cross the boundary
   once with the whole batch" and "cross it once per row" existed
   *entirely inside ZEN's own two documented shapes*, with no algorithmic
   difference in the work performed. Worth stating as design guidance
   anywhere decider2 exposes a call surface a language boundary sits
   behind (the single-record `score()` calling-convention findings in
   EXPERIMENTS.md §N2 are the same lesson from decider2's own side of a
   *Python-internal* boundary — kwargs-binding cost vs. dict-passing cost
   — so this is corroboration of a pattern decider2's own experiments
   already found, not an import).

### Note, but do not prioritize

4. **The trace/simulator shape** (`DecisionGraphTrace`, per-node input/
   output/µs timing, "which rows matched"). Useful for a debugger/UI, but
   decider2's taps (doc 02 §7, ~0.11 ns/row, always-on) plus a table's
   `row_column` path capture already answer "which row matched, with what
   values" for a fraction of the cost of a per-node structured trace —
   confirm coverage is equivalent before spending effort here; likely
   already subsumed.
5. **ZEL's compact interval/comparison cell syntax** (`[a..b)`, `>= x`,
   `in [...]`) as prior art for a possible future *textual* authoring
   shorthand for non-technical rule authors — not a mechanical port
   (decider2's threshold surface is typed pydantic objects, not parsed
   text, and that typing is doing real work), but worth having seen if
   decider2 ever builds a compact one-line-per-condition authoring view.

### Ignore, and why

6. **`functionNode` (QuickJS-per-node).** decider2 has already, deliberately,
   refused this shape (doc 08 §1.1: "config may reference code by
   registered id. It may not contain code"). Adopting it would undo a
   considered position, not fill a gap.
7. **`customNode`/host-adapter dispatch.** Solves "let host code run inside
   the graph," which decider2 already solves, more simply, with
   `step()`/`flow()` composing a plain registered function into the
   pipeline. No gap to fill.
8. **The async-runtime dependency this task's own measurement exposed as
   the *cause* of ZEN's sync-API tax.** Not a feature to import — decider2's
   premise (long-lived compiled process, no Python in the row loop) has no
   async boundary to begin with, so there's nothing here to inherit, and
   deliberately nothing to introduce.
9. **`rand()` / non-determinism in ZEL.** Directly incompatible with doc
   05 §4.2's byte-identical-codegen requirement; not portable at all, by
   design on decider2's side.
10. **General DAG-shaped node reuse for trees specifically.** Genuinely
    more expressive (Q2 said so plainly) but incompatible with the reason
    decider2 trees are single-parent in the first place — doc 05 §7's
    line-cap accounting assumes each node is emitted once. Adopting it for
    trees would undermine the guarantee it would be adopted to avoid
    losing.

(The ZEN expression crate's own architecture — `core/expression/src/`:
separate `lexer/`, `parser/`, `compiler/`, `vm/` directories, i.e. ZEL
compiles to a small bytecode VM rather than tree-walking — is directly
relevant to whether an *interpreted* node evaluator can be fast, which is
exactly the question `experimentation/tree-codegen-vs-interpreted/` is
measuring right now. Flagged here as a pointer for that experiment, not
analyzed further, per this task's scope.)

---

## Recommendation

**Take specific ideas only — bitset table indexing and a `collect` hit
policy — do not adopt ZEN as the execution engine, and do not adopt JDM as
decider2's primary interchange format; treat JDM as a one-way, partial
*export* worth having for governance/visualization, not as a format
decider2's own documents should be authored in or migrated to.** The single
fact that decides it: at the job decider2 actually exists for — a
100,000-row batch — ZEN's own best API is 1.55× slower than decider2's real,
boundary-included `apply()`, not competitive enough to justify replacing a
kernel with no Python in it with a call into a separate Rust runtime; and at
the realtime job, decider2's current, *unoptimized* `score()` already beats
ZEN's own documented entry point by 3.6×. Everything else in this note —
JDM's genuinely good subset-converter story, the two API-shaped surprises in
Q1, the bitset-index and `collect` ideas from Q3 — is real and worth acting
on piecemeal, but none of it clears the bar of "replace the kernel" or
"replace the schema."

## What surprised me

- **The FFI call was never the problem; the Python binding's per-call setup
  was**, and it was possible to isolate exactly how much (a no-op
  passthrough graph costing the same 1.2 ms as the real 12-row table) and
  exactly how to avoid most of it (`async_evaluate` inside a persistent
  loop, 13–17× cheaper) without reading a line of Rust.
- **A converter that refuses is more valuable than one that succeeds.** The
  Turnover table's "row 1 is closed on both ends" refusal, backed by a live
  ZEN evaluation showing the exact value (`1,000,000`) where a naive
  conversion would have silently disagreed, is a stronger Q2 result than
  either of the tables that converted cleanly.
- **decider2 trees are stricter than they look** — genuinely single-parent,
  enforced at build time, not just by convention — which only became
  visible by trying to build a DAG-shaped tree and reading the exact error.

## Files

- `q1_benchmark/rules.py`, `build_decider2.py`, `build_zen.py`, `bench.py`,
  `results.jsonl` — Q1's rule set, both builders, the harness, raw numbers.
- `q2_converter/jdm_to_decider2.py`, `run_table_conversions.py` — JDM ->
  decider2, run against `fixtures/table.json` and every table in
  `fixtures/credit-analysis.json`.
- `q2_converter/example_tree.py`, `decider2_tree_to_jdm.py`,
  `run_tree_conversion.py`, `example_tree.jdm.json` — decider2 Tree -> JDM,
  the 72-combination cross-check.
- `fixtures/` — real JDM documents pulled from
  `github.com/gorules/zen/tree/master/test-data` (`table.json`,
  `credit-analysis.json`, `switch-node.json`, `passthrough.json`), used
  as-is, not authored for this task.
