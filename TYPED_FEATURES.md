# Typed tree features — what was wrong, what changed, what it costs

**Recommendation: merge.** The one number: a realistic 40-leaf, 16-feature
credit tree now runs at **140–156 ns/row end to end against 249–255 before**
(1.6–1.8x faster), while answering the case that used to be silently wrong.
The typed representation was expected to cost a few percent; it turned out
that removing the one-float64-array shape also let the whole walk inline into
the per-row loop, and that is worth more than the discriminator costs.

Branch `worktree-agent-a47fc4a23192c7421` on top of `feature/decider-v2`
(`f90488c`). Full suite: **563 passed, 0 failed** (was 549 green; 14 new
tests in `decider2/tests/test_typed_features.py`). Every number below is from
`decider2/evaluation/typed-features/measurements.jsonl`, appended as it was
measured, with `decider2.__file__` printed on every run so the worktree copy
(not the pip-installed main tree) is the one measured.

---

## 1. What was wrong

A decision tree in `decider2` reads its features through one hot path:
`decider2.trees.interpreter.walk_tree`. Before this change every feature the
tree read — an income in cents, a bureau score, an `is_staff` flag, a string
matcher's result — was coerced into **one `float64` array** per row, and every
threshold into one `float64` tuple. The tree's own `Input` declarations said
`float` for everything, so the boundary layer (`runtime/invoke.py`) cast the
whole column to float64 before the kernel ever saw it.

`float64` has 53 bits of mantissa. Every integer above 2⁵³ = 9 007 199 254 740 992
has a neighbour it cannot distinguish from. Verified on the unmodified tree:

```
Int64 column [9007199254740992, 9007199254740993], node "n == 9007199254740992"
    expected  [1, 0]      (only the first row equals the threshold)
    got       [1, 1]      (both collapse to 9007199254740992 in float64)
```

Two things make this worse than a rounding nit:

- Doc 03 §1.2 **mandates scaled int64 cents for money** precisely so that
  values reconcile exactly with a ledger. A tree over money columns is the
  common case, and the representation quietly threw the mandate away.
- It fails **silently, with a plausible answer** — doc 03 §2.1's "wrong-answer
  bug with no signal … the worst failure mode the design can have". Nothing
  raised; the second row simply took the wrong branch.

Two smaller consequences of the same shape: a `uint32`/`int32` category code
rode in an 8-byte float slot, and — because nothing knew a feature's type at
build time — `sector < 5` on a string column compiled happily and compared
**dictionary codes**, which have no order at all.

## 2. What changed

Three files carry the design; the rest is plumbing.

### 2.1 The row a tree reads is six typed arrays, and the type is program data

`decider2/types.py` gains `FeatureKind` — `F64=0, I64=1, BOOL=2, CODE=3,
STR=4` — and a `Step.typed_args` flag. A typed step's `fn(args, params)`
receives:

```
args   = (f64[:], i64[:], bool[:], int32[:], str_spans int64[:], str_bytes)
params = (float thresholds..., ), (int thresholds..., )
```

Every tree node row now carries **`feat_kind` beside `feat_idx`** (a separate
`int32` array — §4.3 for the alternative). A node says "read slot 7 of the
int64 array"; the same field picks which threshold tuple `thr_slot` indexes,
so an int64 feature is compared against an **int64 threshold** — comparing it
against a float64 threshold would promote the int and reintroduce the defect
one operand over. Booleans are tested as booleans; a bool threshold rides in
the int tuple as 0/1.

Slot numbering has one rule, shared by producer and driver: input `k` of kind
`K` is slot `j` of `K`'s array, where `j` counts the kind-`K` inputs before it
in `Step.inputs` order. Both sides derive `j` from `inputs` alone
(`compile/driver._typed_layout`), so they cannot disagree.

### 2.2 Where a feature's kind comes from

`tree_module(tree, feature_types={"income_cents": int, "is_staff": bool})` —
Python types, or decider 1's polars spellings (`"Int64"`, `"Boolean"`,
`"String"`), or a polars dtype object. Anything not declared is **inferred
from use**: `is_true`/`is_false` only → `bool`; `string_match` → `str`;
anything else → `float`. That last rule is exactly the previous behaviour, so
**an undeclared document encodes as it always did** (pinned by
`test_an_undeclared_tree_encodes_exactly_as_before` and the 50 ported
decider 1 tests, which pass unchanged). A declared name the tree never reads
is an error, per doc 03 §2.2 (an unbound name is a typo).

Kinds are settled by a cheap **probe walk** before the real encode
(`encode._infer_kinds`): the same `encode()` methods run once with every
position discarded, recording only what each feature was asked to do. The
real walk then knows every kind up front, so every slot position is stable
the instant it is handed out — which matters because a computed feature's
expression closure captures its float-array positions at compile time,
during the walk. The wire format is untouched; the node classes in
`schema.py` changed by one keyword argument (`feat_idx=` threaded into
`threshold_slot`) and one adapter line.

### 2.3 What is now rejected at build time

Knowing the kind is what lets the encoder refuse what the single array could
never see. Each is a `ValueError` naming the tree, node and feature:

| tree says | on a feature that is | error |
|---|---|---|
| `<`, `<=`, `>`, `>=`, `==`, `!=`, `between`, `isin` | `str` (categorical) | "a dictionary code has no order and no numeric meaning" |
| any threshold comparison | `bool` | "test a boolean with `is_true`/`is_false`" |
| `is_true` / `is_false` | `str` | "a code has no truth value" |
| `string_match` | `int`/`float`/`bool` | "needs a str column" |
| threshold `5000.5` | `int` | "not an integer" |
| a computed expression reading it | `int`/`bool`/`str` | "cannot read a typed column without widening it" |
| one `InputRef` against an `int` feature and a `float` one | — | "one kernel argument of one type" |

The first row is the correctness win the brief asked about: `sector < 5`
used to compile and compare codes; it is now a build error.

### 2.4 The driver

`compile/driver.py` gets a typed gather beside the existing one: per-kind
homogeneous tuples of whole-column arrays (each padded with a zero-length
read-only dummy when empty — an empty tuple has no element type numba can
index at runtime), five hoisted row buffers refilled per row, one tuple built
above the loop. Tables and Branch/Loop keep the float64 gather untouched;
the flag is opt-in. `interpreted`/`stepped` build the identical six-tuple in
Python so `path_fn` compiles one specialisation shared by every mode.

## 3. What it costs — and why the sign flipped

Measured with `evaluation/typed-features/bench_typed_features.py`: a 40-leaf,
depth-7, 16-feature tree with `<`/`>=`/`==`/`between`/`is_true` nodes, 200k
random rows, median of 9 whole-frame runs after two warm-ups, before and
after run **minutes apart on the same box** (the "before" from a read-only
`git archive` of `f90488c`, `decider2.__file__` printed to prove it). Both
variants produce byte-identical path columns (sha256 prefix in the jsonl).

| tree | metric | before (ns/row) | after (ns/row) | ratio |
|---|---|---|---|---|
| single (16 × Float64) | `apply()` end to end | 222–224 | 125–142 | 0.56–0.64 |
| single | path segment only | 200–203 | 106–118 | 0.53–0.58 |
| mixed (10 Float64 + 4 Int64 + 2 Boolean) | `apply()` end to end | 249–255 | 140–156 | 0.55–0.63 |
| mixed | path segment only | 213–222 | 117–135 | 0.53–0.63 |

Ranges span the two after-runs and two before-runs; the shared box drifts
by ~10%, the effect is ~40–45%.

**This is not what the first cut measured.** The first typed implementation —
a real call per row into `path_fn` carrying six arrays — came in at **441–464
ns/row, 2x slower** (`measurements.jsonl` lines 5–8, labelled
`after-first-cut`). Chasing that produced the finding that drives the whole
result. The table below is a scratch probe over the same built tree and
frame, reproduced three times with the same ranking each time; the numbers
are the last run:

| per-row loop shape | ns/row |
|---|---|
| tuple of six row arrays built per row, call `path_fn` → `walk_tree` (first cut) | 430–517 |
| same, tuple hoisted above the loop | 359–372 |
| `path_fn(args, params)` layer, six arrays, `walk_tree` called inside | 299–341 |
| **old shape**: one float64 array through the same layer | 178–213 |
| kernel calls `walk_tree` **directly**, program arrays as *globals* | 199–202 |
| kernel calls `walk_tree` directly, program arrays passed as *arguments* | 324 |
| **whole chain `inline="always"`** (`path_fn` and `walk_tree`) | **116** |

The cost was never the discriminator. It was **arrays crossing a real call
boundary per row**: each numba array is seven scalars, six of them plus the
program arrays is ~90 scalars spilled to the stack per call, and the callee
reloads every data pointer per node. One float64 array crossed that boundary
for nearly free, which is why the old shape never noticed. Inlining `path_fn`
and `walk_tree` into the driver's per-row kernel at numba-IR level
(`@njit(inline="always")`) makes the row buffers live in registers and the
program arrays compile-time constants, and beats the old shape by a third.
So the honest statement of the price is: **the six-array representation
costs ~130 ns/row if it is ever passed through a non-inlined call; the
production code never does, and pays nothing.** Anyone adding a layer
between the kernel and the walker should re-run the benchmark.

Two things this does *not* change: `interpreted`/`stepped` still call
`path_fn` standalone from Python (its own specialisation compiles and is the
one the disk cache holds — §4.1), and the computed-feature path (`literal_
unroll` over expression closures) is left un-inlined as before; it was and
is the slow, rare path.

## 4. Constraints from the brief, checked

### 4.1 Disk caching, cold then warm, two processes

`NUMBA_CACHE_DIR` persistent, `NUMBA_DEBUG_CACHE=1`, a mixed int/bool tree
run in `fused` and `stepped` in a fresh process, twice
(`test_a_typed_tree_path_fn_is_a_genuine_cache_hit_in_a_fresh_process`, and
the same run by hand):

```
cold  saved : path_fn, out_fn, _fill_array ×4 specialisations, _fill_spans,
              _nonempty, _nonempty_i, compare ×4          loaded: (none)
warm  saved : (none)
      loaded: the same 13 entries
```

`walk_tree` itself no longer has a cache entry: it is inlined into each
tree's `path_fn`, which is the entry that is cached. `path_fn` keeps
`cache=True` with `inline="always"`; numba accepts both and the warm process
loads it. The `@njit` closure that captures per-tree expression Dispatchers
(the computed-feature `path_fn`) stays **not** `cache=True`, as before — the
brief's "grows the index without hitting" trap.

### 4.2 No arity ceiling; retune never recompiles; equivalence

- `tests/test_no_arity_ceiling.py` passes unchanged: the 400-feature tree is
  400 float slots in one array, gathered by one runtime-indexed loop; a
  mixed tree is the same loop run once per kind. Nothing is keyed on a count.
- `test_retuning_an_int_threshold_never_recompiles`: `count_new_compiles()`
  is 0 across a retune of an int threshold. `params` is a `(floats, ints)`
  pair whose numba type is fixed by the param annotations, not the values.
- `assert_equivalent` (interpreted/stepped/fused, plus `score()`) is green on
  the int64 case, the bool case, the int chain, and a mixed tree with float +
  int + bool + `string_match` + computed feature + `isin` in one document.
- Doc 05 §4.2 determinism: encoding the same tree twice gives identical
  arrays, `feat_kind` included.

### 4.3 `feat_kind`: separate array vs high bits of `feat_idx`

`bench_feat_kind_packing.py`, same tree and frame, two otherwise identical
inlined walkers, interleaved twice: separate array **116.7 / 116.6 ns/row**,
packed high bits **116.4 / 125.7 ns/row** (mins 116.1 vs 115.6). Inside the
box's noise. The separate array is kept: it is what the program data reads
as, and there is no measurable case for the shift-and-mask.

## 5. The raw-string slot (for the two string strands)

`FeatureKind.STR` is reserved and the driver side is built and tested, so a
lazy matcher has somewhere to read actual bytes without another
representation change:

- An input annotated `bytes` is gathered as a **span**: the row's `int64` span
  array holds `(start, end)` for each such feature (`_fill_spans`), read
  straight off polars' `offsets` buffer; the whole-column `values: uint8`
  buffer travels as the sixth tuple member, one entry per string feature.
  Both are exactly what `Series._get_buffers()` returns for a String column
  (`offsets: Int64`, `values: UInt8`), zero-copy — verified.
- The driver reads `registry[name]` as the offsets and
  `registry["__bytes__" + name]` as the bytes, mirroring the existing
  `__valid__` convention.
- `test_the_raw_string_slot_carries_polars_bytes_zero_copy` runs a step that
  returns `len(s)*1000 + first byte` through both the fused kernel and the
  interpreted row path, from real polars buffers.

What is **not** built: the boundary does not yet produce such a column
(`boundary/dtypes.py` dictionary-encodes every String to `CODE`), and
`walk_tree` has no node kind that reads the slot. Adding them is: a
`RawStringPlan` in `dtypes.py` returning `(offsets, values)`; `apply()`
writing both registry names; a `STR_MATCH` node kind in `walk_tree` (the
kind switch is already there) whose matcher compares `sbytes[k][s64[2k]:
s64[2k+1]]` against a pattern table. The int32-code route stays the fast
path for exact matches; the slot exists for what codes cannot express
(prefix/contains/regex, doc 05 §1.5's stated gap).

## 6. What it makes possible, and what it does not do yet

- **Fold the hoisted string matcher into the walk.** A `str` feature's codes
  can now ride in the `int32` row array and be compared against a code
  threshold directly (`CODE` nodes already compile). Not done here because
  `runtime.invoke._resolve_str_param_code` resolves a `str` param against a
  step's *single* str input — a tree with two string features would need
  per-param column binding first. Worth doing: it removes one kernel launch
  and one int64 intermediate per string feature.
- **Typed computed features.** `decider2.expr` closures are float64 over
  `(f64_row, thr_f)`; an expression over an `int`-declared feature is
  refused rather than widened. Money arithmetic in a computed feature
  therefore still needs the feature declared `float` (or a step before the
  tree). Doc 03 §1.2 says accumulate in float64 anyway, so this is
  consistent, but it is a limitation to know about.
- **Boundary casting is still by annotation.** A Float64 column handed to an
  `int`-declared feature is `astype(int64)` at the boundary — truncation, the
  same rule every `-> int` hand-written step already lives under. A typed
  input schema at `build` time (doc 07 §5) is the right place to make that
  an error; it is out of scope here.
- Pre-existing, noted, untouched: `decider2/expr.py`'s per-node closures are
  `@njit(cache=True)` while capturing other Dispatchers (`left_fn`/`right_fn`)
  — the pattern the brief says must not be cached. It predates this work and
  the computed-feature tests pass; flagged for whoever owns `expr.py`.

## 7. Files

- `decider2/src/decider2/types.py` — `FeatureKind`, `feature_kind()`,
  `Step.typed_args`.
- `decider2/src/decider2/compile/driver.py` — typed gather (`_typed_layout`,
  `_typed_input_arrays`, `_typed_params`, `_fill_spans`, `_typed_row_args`,
  `_build_typed_kernel`); `_fill_array` loops over `len(out)`.
- `decider2/src/decider2/trees/interpreter.py` — `walk_tree` switches on
  `feat_kind`, `inline="always"`.
- `decider2/src/decider2/trees/encode.py` — kinds, probe walk, typed
  thresholds, the build-time rejections, `feature_types=`.
- `decider2/src/decider2/trees/schema.py` — `feat_idx=` threaded into seven
  `threshold_slot` calls; `_ExprEmitAdapter.name_index` asks for the float
  slot.
- `decider2/src/decider2/trees/build.py` — `tree_module(feature_types=)`,
  `explain()` prints kinds.
- `decider2/tests/test_typed_features.py` — 14 tests.
- `decider2/evaluation/typed-features/` — the two benchmark scripts and
  `measurements.jsonl`.

## 8. Should it be merged?

Yes. The change fixes a silent wrong answer on the column type the design
mandates for money, rejects a class of meaningless comparisons at build time,
keeps every existing document encoding identically, keeps the wire format,
the arity property, retune-without-recompile and disk caching, and is faster
by a third on the realistic body. The two honest caveats are §3's "never
pass the row tuple through a non-inlined call" (documented at the call site
and in `walk_tree`'s docstring) and §6's refusal of typed features inside
computed expressions, which is a loud error rather than a silent one.
