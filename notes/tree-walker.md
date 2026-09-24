# Trees as data: one compiled walker, one Python twin

`decider.steps.trees` choices that `TreeConfig`, tables and sessions build on.

- **One row node per tree.** `TreeConfig.to_ir` emits a single `row`
  `CallNode` under the config's name. `fn` is `walker.walk`, one njit function
  for every tree. The tree is data (`encode.Program`): a program of
  `(kind, feature, op, threshold, then, else)` rows, literal thresholds, rule
  roots, computed features as postfix rows, string patterns and every
  output's values, packed into one int64, one float64 and one byte array.
  Editing a literal or the tree's shape changes array contents only, and a
  `ParamRef` is a param, so neither recompiles. numba compiles `walk` once
  per *type* of `(row, params, consts)`: feature kinds and counts, the param
  names (the bundle class), output dtypes. About 0.5-6 s per new shape, then
  the disk cache.
- **The consts are addresses, not arrays.** The first version passed the
  arrays: every array crossing the per-row call was reference-counted in and
  out (about 60 atomic NRT calls a row), which cost 525 ns a row on a
  one-node tree. Passing the arrays' addresses and reading them with the
  shim's `load_i64`/`load_f64`/`load_u8` intrinsics made it 30 ns. The
  arrays stay alive because the node's `reference` holds them
  (`reference.arrays`); the node holds its reference.
- **Why rows are laid out float, int, bool, span.** A row node gets its
  inputs as one tuple, and numba can't index a mixed tuple at run time.
  Declaring inputs grouped by kind (then by name) lets a numba `overload`
  split the tuple with static slices into homogeneous tuples. Params are
  ordered the same way. No source is generated: the overload's
  implementation is an ordinary closure over the slice bounds.
- **Outputs are picked by an intrinsic** (`walker.pick`), because the
  output tuple's length and dtypes differ per tree. Its codegen loops over
  the outputs in Python and emits one call per output to `_value`,
  `_nullable` or `_trace`, each compiled once per output type; LLVM inlines
  them. History: numba-level inlining of a type-level recursion re-typed
  every level at every level (3 s at 4 outputs, 48 s at 8); `forceinline`
  recursion was linear but a Python frame chain per output, so a
  `mode: "all"` set of 100 rules hit `RecursionError` (01's 521-rule set
  did). Halving the recursion fixed the error but not the time: LLVM's
  InstCombine/SROA on slices of a 500-field tuple was ~100 s, and runtime
  indexing into such a tuple is quadratic to compile too. The intrinsic
  plus `no_cpython_wrapper` on `walk` (its Python entry unboxed the
  500-spec tuple: 50 s of the rest) compiles a 521-output walk in 25 s,
  128 outputs in 2 s. `tree_walk.py` at load ~9, 2026-09-24: fused score
  p50 58.0 µs (before 57.8-58.1), string-gated 99.8 (100.9-101.5), batch
  188 ns/row (265-275). The walk loop, byte matching and expression
  evaluation are real calls, not numba-inlined; before that a one-output
  tree took 18-34 s to compile.
- **String outputs are `Literal[...]` outputs.** Kernels can't write strings.
  A tree's `String` column is declared `Literal["a", "b", ...]`; `fn` returns
  the value's index (-1 for null) and `engine.compile.units` decodes it into
  a string column after the kernel (`literal_choices`). `reference` returns
  the string itself. This needs no per-runner state and suits decision
  tables too. It is an IR convention that should be ratified in
  `spec-amendments.md`.
- **Strings are matched as bytes at the node.** A string feature is a
  `bytes | None` input. Compiled runners hand a `bytes` column to a kernel as
  `(address, byte length)` spans through the Arrow shim (`extract_frame`),
  length -1 for a null, and a `str` param of such a node as a span of its
  UTF-8 bytes (kept alive with the cached bundle). A kernel column that is a
  2-D `(n, 2)` span table yields one span per row. exact, starts_with,
  ends_with and contains compare bytes against a pattern table in the
  program. For one to 32 records (`score`) the runner encodes the strings
  itself, which beats building a frame to export (about 40 µs saved).
- **regex, case folding and trimming run in Python in every mode.** numba
  has no regex engine and Unicode folding and whitespace need tables a
  kernel doesn't have; decider2 refused them. Here a tree using one gets a
  Python `fn` that numba can't type, so compiled modes run it as a
  `Fallback`, which now gets plain Python values (strings, not codes or
  spans) and stores `str` outputs as objects. Before, a scorecard's string
  bins silently never matched in compiled modes. None of this preprocesses
  strings into dictionary codes.
- **The reference walker is independent.** It walks the `Tree` model node by
  node, calling `visit(node_id)` for every node it passes (leaves included),
  evaluates computed features with `Expr.evaluate` and matches strings with
  Python's `str` methods and `re.search`. It shares only the feature kinds
  with the encoder, so the tests that run every fixture in all three modes
  compare two implementations.
- **Kinds.** Declared in `feature_types`, else inferred as decider2 did:
  `string_match` makes `str`, only `is_true`/`is_false` makes `bool`,
  anything else `float`. An int feature is compared against int64
  thresholds; a fractional literal on it is an error. Computed features are
  float arithmetic and may only read float features.
- **Nulls follow decider_old unless the config asks for strict.** decider_old
  built every tree as polars `when/then/otherwise`, so a comparison with a
  null was null (three-valued logic: NOT keeps it null, AND/OR are Kleene)
  and a null condition took the otherwise branch; a cases node tried its
  next case. `TreeConfig.null_handling="otherwise"` (the default, so every
  decider_old document answers as before) reproduces that;
  `null_handling="error"` makes numeric and boolean features required
  inputs (`MissingInputError`). A string match
  has decider_old's own per-condition `null_handling`: `no_match` (default)
  is a plain false, `match` a plain true (both flip under NOT), and `error`
  makes that string column a required input, since decider_old evaluated
  every condition for every row.
  - **One extra target per program row, no duplicated rows.** A null test
    is "unknown", which only a NOT tells apart from false. Starting from a
    node (unknown goes where false goes) and pushing through AND/OR/NOT,
    unknown always ends at the same place as false, or, under an odd number
    of NOTs, as true. So each comparison row carries an `UNKNOWN` target
    (its `else` or its `then`) and the walker jumps there on a null; no
    Kleene state is carried and no subtree is encoded twice.
  - **A null number is a fill, not an Optional.** A float feature is
    `missing_as(NaN)` and an int feature `missing_as(-2**63)`; the walker
    treats NaN and that sentinel as unknown, so a computed feature over a
    null (NaN) is unknown too, and the Python walker reads both as `None`.
    Declaring them `T | None` instead cost one mask array per feature per
    launch: on the 18-feature benchmark tree `score()` p50 went from 63 to
    102 µs (to 75 µs with one shared all-valid mask); with fills it is back
    at 58-61 µs. Back to back at load ~3: before 550 ns/row and p50 64 µs,
    after 490 ns/row and 56 µs (decider2: 539-611 ns/row, 200-218 µs). A
    NaN in the data is therefore a null. Bool features have
    no spare value, so they stay `bool | None` (`split` groups an
    `Optional` by its base type).
- **Outputs are laid out per leaf, not per output row.** A LEAF program
  row holds the leaf's ordinal (-1 for a `result_idx: -1` leaf and a missing
  branch), and each output's values repeat the row every leaf selects. Then
  the index a walk ends at also names the leaf, so `path_output` (the leaf
  id as a `Literal` String output, per rule in `mode: "all"`) is one more
  output with values `0..n-1`, and the walker didn't change at all. The
  cost is a few more values when several leaves share a row. A leaf that
  selects the default row reports a null path, as the default answered.
- **`trace_output` is a path number, not a list.** A kernel can't write a
  variable-length list or a string, and the leaf doesn't name the path once
  a node has two parents. So the encoder numbers paths Ball-Larus style:
  each node's path count is the sum of its branches' (a missing branch is
  one path), and the jump into branch `i` weighs the counts of branches
  `0..i-1`. Every program row carries three weights, one per target
  (`W_THEN`, `W_ELSE`, `W_UNKNOWN`: nonzero only on jumps that leave a
  node), and `_walk` sums the weights of the jumps it takes, so the sum is
  the path's index among its root's paths. The output is a `Literal` of
  every path's `>`-joined node ids, enumerated in that order, which the
  unit decodes like any string output; no new output kind reached the
  engine. First-match adds each rule's base (the paths of earlier rules,
  stored after the roots) and reports the last rule walked. Costs: the
  walker does one more add per row whether traced or not (`tree_walk.py`
  back to back at load ~7, 2026-09-24: fused score p50 63.4 -> 62.5 µs,
  string-gated 112.4 -> 111.0 µs, batch 226 -> 233 ns/row, all within
  noise), and a tree gets one string per path, so a DAG
  whose shared nodes multiply paths past 100k is refused. A node that
  compiles to no rows (a condition that tests nothing) can't be told apart
  from its child, so tracing one is an error.
- A numeric output column holding `None` is declared `T | None`.
- **`mode: "all"`** writes `<rule name>.<column>` per rule (`rule_<i>` when
  unnamed), since a row node writes flat columns, not decider_old's structs.
- **Expressions evaluate both sides of `and`/`or`**, so a guard such as
  `x == 0 or 1 / x > 2` raises where Python short-circuits.

**Measured** (`benchmarks/trees_vs_decider2.py`: a 127-node tree over 18
features with a String, a Float64 and an Int64 output; 1M rows through
`run(df)`, 20k `score(dict)` calls; dev box under load average ~15,
2026-09-23):

| tree | engine | ns/row | score p50 | score p99 |
|---|---|---|---|---|
| numeric | decider2 fused | 748 | 209 µs | 375 µs |
| numeric | decider fused | 592 | 58 µs | 78 µs |
| string-gated | decider2 fused | 614 | 257 µs | 326 µs |
| string-gated | decider fused | 689 | 101 µs | 117 µs |

decider2's batch time includes its `decode()` of the String column. The
string-gated run adds a `starts_with` node on a `channel` column in front of
the same tree.
