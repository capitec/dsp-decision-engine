# PORTED.md — decider 1 -> decider2 tree/table conformance port

Ledger for the migration-conformance port of decider 1's decision-tree and
decision-table unit tests into decider2. Source: `tests/rules/
test_conditions.py`, `tests/rules/test_tree_end_to_end.py`, `tests/rules/
test_tree_migration.py`, `tests/rules/test_parameters.py`, `tests/credit/
decision_table/test_decision_table.py` (65 tests total). Target: this
directory's `test_trees_ported_conditions.py`, `test_trees_ported_end_to_
end.py`, `test_trees_ported_migration.py`, `test_trees_ported_parameters.py`,
`test_tables_ported.py` (50 test functions total).

## Headline finding: a real decider2 bug, found and fixed

Porting `test_composite_and_or_not` (the plainest possible AND/OR/NOT
composite test decider 1 has) surfaced a genuine correctness bug in
`decider2/trees/codegen.py`. It is now fixed; see "Bug found and fixed"
below before reading the rest of this ledger — it is the most valuable
single result of this task.

## How to read this table

- **Ported** — same config, same expected answer as decider 1, unchanged.
- **Ported (adapted)** — same semantic intent and expected answer, but the
  construction had to change because the *vocabulary* differs mechanically
  (decider 1 nests rules with `then=`/`otherwise=`; decider2 is a node/edge
  graph) or because a supporting mechanism has a different shape (see
  "Adaptations" below).
- **Ported (divergence)** — decider2's actual, reproducible behaviour for
  this config is DIFFERENT from decider 1's, so the ported test asserts
  decider2's real behaviour instead of decider 1's answer. Every one of
  these is written up under "Divergences found" below.
- **Not ported (GAP)** — no decider2 construction expresses the config at
  all; recorded, not faked. Every one is a listed KNOWN GAP or a clearly
  decider-1-internal implementation detail with no decider2 analogue.

## `test_conditions.py` -> `test_trees_ported_conditions.py` (19 -> 17 functions)

| decider 1 test | Status | Notes |
|---|---|---|
| `test_all_numeric_comparison_operators` | Ported | direct |
| `test_between_variants` | Ported | direct |
| `test_null_and_boolean_operators` | Ported (partial) | `is_true`/`is_false` half ports. `is_null`/`is_not_null` half is GAP — `UnaryIsNull`/`UnaryIsNotNull` have no decider2 node; null policy is declared in the signature (doc 03 §1's four tiers), not tested per-condition. |
| `test_string_match_types` | Ported (partial + divergence) | `match_type="exact"` ports directly. `contains`/`starts_with`/`ends_with`/`regex` port as `UnsupportedInKernel` assertions — decider2's actual, deliberate refusal (doc 05 §1.5: a kernel sees an int32 code, not text). |
| `test_string_match_case_insensitive_and_trim` | Ported (divergence) | Both `case_sensitive=False` and `trim_whitespace=True` port as `UnsupportedInKernel` assertions, same reason. |
| `test_cases_ranges_lower_and_upper_inclusive` | Ported | direct |
| `test_cases_isin` | Ported | direct |
| `test_cases_string_match` | Ported (divergence) | decider 1's config groups strings by `match_type="starts_with"` prefix; no `exact`-only config reproduces the same groupings, so this ports as an `UnsupportedInKernel` assertion instead of a same-answer test. |
| `test_composite_and_or_not` | Ported | direct — **this is the test that found the codegen bug; see below.** |
| `test_nested_composite` | Ported | direct (also exercises the fixed bug, nested one level deeper) |
| `test_default_returned_when_no_match` | Ported | direct |
| `test_correct_output_row_indexed` | Ported | direct |
| `test_special_numeric_values_through_range_rules` | Ported (partial + divergence) | `inf`/`-inf`/`nan` cases port unchanged (none is a polars null). The `None` case does not port with the same assertion — see "Divergences found" §1 — and is replaced by `test_null_input_is_routed_away_not_evaluated`. |
| `test_prioritized_first_match_wins` | **Not ported (GAP)** | `PrioritizedFlatRuleModule` — no decider2 equivalent. |
| `test_prioritized_falls_back_to_default` | **Not ported (GAP)** | same |
| `test_unary_is_in` | Ported | direct |
| `test_cases_ranges_inputref_bounds` | Ported (adapted) | decider 1 gave the `InputRef` its default via `ParameterInfo(default_value=...)` at the module boundary; decider2 has no such thing — the default is supplied at `tree_module(tree, params={...})` instead (doc 03 §4.3's pre-bind). Same values, same answer. |
| `test_cases_string_match_inputref_pattern` | Ported (adapted) | same adaptation |
| `test_prioritized_module_with_parameters` | **Not ported (GAP)** | `PrioritizedFlatRuleModule` |

Plus one new test with no decider-1-test counterpart, added to document the
divergence found while porting the `None` case above:
`test_null_input_is_routed_away_not_evaluated`.

## `test_tree_end_to_end.py` -> `test_trees_ported_end_to_end.py` (29 -> 18 functions)

| decider 1 test | Status | Notes |
|---|---|---|
| `test_nested_unary_then_cases_ranges` | Ported | direct (also ported independently in the prior session's `test_trees_migration.py`; kept here too — this file's job is a complete port of its own source file) |
| `test_nested_cases_ranges_then_unary` | Ported | direct |
| `test_three_level_nested_tree` | Ported | direct |
| `test_composite_inside_nested_tree` | Ported | direct |
| `test_prioritized_all_mode_returns_each_rule_independently` | **Not ported (GAP)** | `PrioritizedFlatRuleModule`/`PrioritizationMode.all` |
| `test_prioritized_all_mode_all_rules_default` | **Not ported (GAP)** | same |
| `test_path_unary_then_branch` | Ported (adapted) | decider 1's path is a STRING built from a custom `output_fn`/`branch_stack` ("score,0"). decider2's `<name>_path` (doc 03 §7) is the terminal leaf's `result_idx` — one int, not a breadcrumb. Ported to assert on that int where it distinguishes the same outcomes. |
| `test_path_otherwise_branch_index_is_branch_count` | Ported (adapted) | decider 1's "otherwise" path index is `len(conditions)` (a branch-count convention). decider2's is the `-1` default sentinel (`LeafNode`'s own convention). Ported to assert decider2's actual convention. |
| `test_path_nested_tree_depth_reflects_decisions` | Ported (partial) | The "path string gets longer with nesting depth" assertions do not port — decider2's path is terminal-only, so "depth" isn't a concept it has. The answer-correctness and per-leaf `_path` value assertions do port. |
| `test_path_cases_ranges_records_correct_bucket_index` | Ported (adapted) | decider 1's path index for a `CasesRanges` match already equals the bucket position, same as decider2's `result_idx` — nearly a direct port, just int not string. |
| `test_path_identical_inputs_always_produce_identical_paths` | Ported | direct, via `<name>_path` |
| `test_path_null_input_takes_otherwise_path` | **Not ported (same-assertion)** | Same divergence as `test_conditions.py`'s special-values `None` case — see "Divergences found" §1. Replaced by `test_trees_ported_conditions.py::test_null_input_is_routed_away_not_evaluated`. |
| `test_multi_column_output_all_fields_correct` | Ported | direct |
| `test_multi_column_output_default_row_on_no_match` | Ported | direct |
| `test_multi_column_output_in_prioritized_first_match` | **Not ported (GAP)** | `PrioritizedFlatRuleModule` |
| `test_build_parameters_expr_no_runtime_column_uses_defaults` | **Not ported (GAP)** | Unit-tests decider 1's *internal* `flat_rules.impl.build_parameters_expr` function directly. decider2's parameter resolution is a different mechanism entirely (kernel `param()` args, resolved in `runtime/invoke.py`); there is no analogous internal seam to test. |
| `test_build_parameters_expr_empty_schema_returns_none` | **Not ported (GAP)** | same |
| `test_build_parameters_expr_parameter_with_no_default_uses_runtime` | **Not ported (GAP)** | same |
| `test_prioritize_results_empty_list_returns_default` | **Not ported (GAP)** | Tests decider 1's internal `prioritize_results`, part of the `PrioritizedFlatRuleModule` machinery (GAP). |
| `test_optimized_execution_path` | **Not ported (GAP)** | `use_optimized_execution=True`/`OptimRunPolarsExpression` is decider-1-internal execution-mode plumbing with no decider2 concept — every decider2 tree is already a compiled numba kernel; there is no "optimized" vs. "unoptimized" mode to select. |
| `test_optimized_execution_prioritized` | **Not ported (GAP)** | same, plus `PrioritizedFlatRuleModule` |
| `test_run_polars_expression_execute` | **Not ported (GAP)** | Tests decider 1's internal `RunPolarsExpression.execute()` class directly; decider2's compiled-kernel model has no polars-expression-tree intermediate to test the same way. The BEHAVIOUR this pins (build once, execute correctly) is already covered by every other ported test that calls `tree_module()` once and `.apply()`s it. |
| `test_cases_ranges_empty_conditions_returns_otherwise` | Ported (divergence) | See "Divergences found" §2. |
| `test_cases_string_match_empty_conditions_returns_otherwise` | Ported (divergence) | same |
| `test_cases_isin_empty_conditions_returns_otherwise` | Ported (divergence) | same |
| `test_composite_rule_empty_conditions_evaluates_false` | Ported (divergence) | See "Divergences found" §3. |
| `test_cases_ranges_get_required_parameters_with_inputref_bounds` | Ported (adapted) | decider 1's `get_required_parameters()` -> decider2's `required_params()` (same idea, every node type, `trees/schema.py`). |
| `test_cases_string_match_get_required_parameters_with_inputref` | Ported (adapted) | same |
| `test_cases_isin_get_required_parameters_with_inputref` | Ported (adapted) | same |

## `test_tree_migration.py` -> `test_trees_ported_migration.py` (6 -> 4 functions)

| decider 1 test | Status | Notes |
|---|---|---|
| `test_v1_parses_and_upgrades_to_v3` | **Not ported (GAP)** | decider 1's v1 tree format and its v1->v2->v3 upgrade chain (`V1Tree`, `.upgrade()`) have no decider2 equivalent at all. KNOWN GAP, explicitly listed in the task brief: "decider 1's v1 `Tree` format — superseded; v2 ports via `from_v2_range()`". |
| `test_v1_tree_as_basemodule_produces_correct_output` | Ported (adapted) | decider 1's own assertion only exercises the plain-numeric subtree of its v1 fixture (the string-match subtree is separate and unasserted). That subtree is expressible directly as a v3 `Tree` with the identical answer, so it ports that way — same intent, not the same document. |
| `test_v3_numerical_and_range_nodes_execute_correctly` | Ported | direct |
| `test_v1_tree_parse_rejects_missing_nodes` | Ported (adapted) | `V1Tree.model_validate` -> decider2's `Tree.model_validate` (no v1 format exists here); same "required field absent" `pydantic.ValidationError`. |
| `test_v1_tree_parse_rejects_unknown_node_type` | Ported (adapted) | same adaptation |
| `test_can_create_default_tree` | **Not ported (GAP)** | decider2 has no `Tree.default_tree()` factory (decider 1's `Tree` there is an unrelated versioned wrapper this package never built). |

## `test_parameters.py` -> `test_trees_ported_parameters.py` (6 -> 6 functions)

| decider 1 test | Status | Notes |
|---|---|---|
| `test_inputref_uses_default_when_no_runtime_column` | Ported (adapted) | `ParameterInfo(default_value=...)` -> `tree_module(tree, params={...})`. Same default, same answer. |
| `test_inputref_uses_runtime_struct_column` | Ported (adapted, renamed `test_inputref_uses_runtime_override`) | decider 1's fixture used ONE override value for every row, so despite the mechanism being different (a per-row struct column vs. a per-call `params=` kwarg), the two happen to produce the identical answer for this specific config. |
| `test_inputref_runtime_overrides_default_per_row` | **Not ported (same-assertion)** | See "Divergences found" §4. Replaced by `test_inputref_cannot_vary_per_row_only_per_call`. |
| `test_inputref_between_with_two_parameters` | Ported (adapted) | decider 1's fixture never actually overrides at runtime (no `parameters` column in its frame) — pure rename of the default-supplying mechanism, same as row 1. |
| `test_computed_feature_two_column_expression` | Ported (renamed `test_computed_feature_two_column_expression_compiles`) | `Feature(type="computed", ...)` is admitted again (`decider2.expr`, doc 08 §1.1/§3.2, doc 06 §O15) — now gives the SAME answer decider 1 gave, compiled to numba source at build time instead of evaluated with `simpleeval` at runtime. |
| `test_computed_feature_uses_parameter` | Ported (divergence, renamed `test_computed_feature_p_dot_attribute_syntax_is_refused`) | decider 1's `p.bonus` attribute-access convention for a parameter inside the expression string does not carry over — `decider2.expr` rejects attribute access unconditionally. This is the one genuine remaining GAP; ported as a `pydantic.ValidationError` assertion naming the construct, not `ComputedFeatureRemoved` (no longer raised for anything). |

## `test_decision_table.py` -> `test_tables_ported.py` (5 -> 5 functions)

All five port directly, unchanged config and answer — the decision-table
vocabulary was kept identical across the migration (`decider2/tables/
schema.py`'s own docstring says so). Only the harness differs mechanically:
`module({"input": df})` + `.struct.unnest()` becomes `table_module(...)` +
`built.decode(flow(built.module).apply(frame, shared=built.shared))`.

| decider 1 test | Status |
|---|---|
| `test_between_maps_ranges_to_output_labels` | Ported |
| `test_between_default_when_outside_all_ranges` | Ported |
| `test_in_expression_categorical_lookup` | Ported |
| `test_and_expression_requires_all_conditions` | Ported |
| `test_multiple_output_columns_all_populated` | Ported |

(This duplicates the five decision-table tests already ported, under
different names, in the prior session's `decider2/tests/test_trees_
migration.py` — intentional, per the task's explicit target file list.)

---

## Bug found and fixed

**`test_composite_and_or_not` (`test_trees_ported_conditions.py`) failed on
first attempt** — the single most valuable outcome of this task, per its
own instructions.

**Config:** a `CompositeNode(op="and", conditions=[UnaryGreaterThan(feature=
"x", threshold=5.0), UnaryLessThan(feature="x", threshold=10.0)])`, i.e.
`5 < x < 10`.

**Input:** `x = [3.0, 7.0, 12.0]`.

**decider 1's answer:** `["no", "yes", "no"]` (only `x=7` is between 5 and
10).

**decider2's answer (before the fix):** `["no", "no", "no"]` — `x=7` was
wrongly rejected.

**Root cause:** `decider2/trees/codegen.py`'s `_Emitter._unary_test`
names a threshold's kernel `param()` from `f"{node_id}_{role}"`, where
`role` is a constant per operator shape (`'thr'` for the six primitive
comparisons, `'min'`/`'max'` for `Between`, `'isin_0'` for `IsIn`). That is
correct for a lone `UnaryNode`, where `node_id` alone is already unique.
But `_condition_test`, which walks a `CompositeNode`/`CompositeCondition`'s
`conditions` list, called `_unary_test` with only `node_id` — no
per-position index — so every condition in ONE composite's list that
shared an operator shape collided on the SAME parameter name. Here, both
`UnaryGreaterThan` and `UnaryLessThan` use `role='thr'`, so both compiled
to reading the ONE argument `root_thr`. The emitted body was literally
`if (x > root_thr) and (x < root_thr): return 0` — never satisfiable for
any `x`, so the composite silently always evaluated `False`. No error,
no warning: silently wrong answers for a bread-and-butter config (AND of
two range bounds on one feature) that the existing decider2 test suite
happened never to exercise (its one `CompositeNode` test,
`test_trees.py::test_composite_and_between_and_isin_nodes`, combines a
`Between` with an `IsTrue`, which don't share a role and so never
collided).

**Classification:** genuine decider2 bug, not a documented divergence and
not a decider-1 bug being fixed. Confidence is high: reproduced from first
principles by inspecting the emitted source directly (`emit_tree(tree).
source` showed `root_thr` used for both `>` and `<`), and the fix was
verified against three independent configs — two plain `UnaryGreaterThan`+
`UnaryLessThan` conditions, a `CompositeCondition` nested two levels deep
with the same shape repeated, and two `UnaryBetween` conditions on
different features in one `OR` — plus the full existing 356-test decider2
suite, all passing unchanged after the fix (no test anywhere asserted the
old, colliding parameter names).

**Fix (exact change):** `decider2/trees/codegen.py` — `_unary_test` and
`_condition_test` both gained an optional `cond_idx: str | None = None`
parameter, defaulting to `None` (which reproduces the OLD role names
exactly — `'thr'`, `'min'`, `'max'`, `'isin_0'` — so every existing
single-condition `UnaryNode` is byte-for-byte unaffected). The call site
inside `_emit_node` for a `CompositeNode`'s own `conditions` list now
passes `str(i)` for each condition's position
(`self._condition_test(c, node_id, str(i)) for i, c in enumerate(data.
conditions)`), and `_condition_test`'s own recursion into a nested
`CompositeCondition` extends that index path (`f"{cond_idx}_{j}"`) rather
than resetting it, so a composite nested three levels deep still names
every leaf threshold uniquely (e.g. `root_thr_0`, `root_min_1`,
`root_max_1`). Full existing decider2 suite (356 tests) still passes
unchanged.

---

## Divergences found (documented, not fixed — see each for why)

### 1. A `None` in a REQUIRED-tier tree feature is routed away, not evaluated

decider 1's numeric comparisons run directly over a polars `DataFrame`,
including any nulls — a null simply fails every comparison and falls
through to `otherwise`/default. Two decider 1 tests depend on this exactly
(`test_special_numeric_values_through_range_rules`'s `None` case,
`test_path_null_input_takes_otherwise_path`).

decider2 declares a tree feature's kernel-function parameter as plain
`float` (no `| None`) unless told otherwise — doc 03 §1 tier 1,
`NullPolicy.REQUIRED`, "routed, not raised, by default". Confirmed by
direct investigation of `decider2.boundary.extract_frame`: a `None` in
such a column routes the WHOLE ROW to a `Decision` (default
`Decision.REFER`) at extraction, before the tree ever runs at all. This is
a materially different mechanism from decider 1's "the comparison reads
null and returns False" — not just a renamed version of the same thing.

It is also, right now, not fully observable through `.apply()`'s output
frame: `runtime/invoke.py::_scatter_back`'s own docstring admits the
routed row's terminal/`_path` column is filled with an arbitrary
placeholder (`nan` for a float terminal, `0` for an int64/bool terminal —
`_path` is int64) rather than a real null, "not a claim that 0/False is
the routed row's real answer... Closing that gap for real means
`DtypeGroup` growing an optional validity array." That reads as a
deliberate, acknowledged scope boundary (rendering the actual `Decision`
is explicitly left to `observe/`, which doesn't exist yet), not an
accidental bug — so it was not touched. Both tests were replaced with
`test_null_input_is_routed_away_not_evaluated`, which asserts the real,
documented mechanism (`extract_frame`'s `NullRouting`) directly rather
than pinning the admitted placeholder value.

**Is this a decider2 bug?** No — confidently not. It is doc 03 §1's
routing design working as specified; the only rough edge (the placeholder
terminal value) is explicitly flagged in the source as an intentional,
narrow scope cut, not an oversight.

### 2. An empty-conditions `Cases*` node: schema accepts it, build rejects it

decider 1 defines `CasesRanges`/`CasesStringMatch`/`CasesIsIn` with zero
conditions as legal and meaning "always `otherwise`" (three decider 1
tests pin exactly this). decider2's schema also accepts the shape
(`arity == len(conditions) + 1 == 1` when empty, i.e. exactly the one
"otherwise" edge) — but `tree_module()` then rejects it at BUILD time with
`ValueError: ... parameter 'x' is declared but never referenced in the
body`, because `CasesRanges.required_features()` still reports the node's
feature as a required kernel argument even though the (correctly) empty
generated body never reads it.

**Is this a decider2 bug?** Plausibly a small one, but not fixed here —
it needs a design call this task isn't positioned to make unilaterally:
either (a) make the schema reject an empty `conditions` list outright,
matching `CompositeNode`'s own explicit validator (see divergence 3,
immediately below, which already does exactly this and produces a much
clearer error at construction time instead of a confusing one at build
time), or (b) special-case `required_features()`/codegen so an empty
`Cases*` node legitimately builds and always takes `otherwise`, matching
decider 1. Recommend (a) — it is a two-line validator matching a pattern
that already exists next to it — but that is a judgement call worth a
maintainer's sign-off, not something to change silently while porting
tests. Ported as the actual, reproducible current behaviour (`pytest.
raises(ValueError, match="never referenced")`) so the finding survives in
the suite either way.

### 3. `CompositeNode`/`CompositeCondition` with zero conditions: rejected at the schema, not evaluated as "always False"

decider 1's `CompositeRule(op=AND, conditions=[])` is legal and defined to
always evaluate `False` (its own implementation's choice, not a
mathematical truth — vacuous AND is conventionally `True`).

decider2's `CompositeNode`/`CompositeCondition` both raise
`pydantic.ValidationError` — "composite node needs at least one
condition" — at construction, via an explicit `model_validator`. This is
arguably the RIGHT fix for divergence 2 above too (see that entry). Ported
as the real, deliberate decider2 behaviour, not decider 1's "always
False".

**Is this a decider2 bug?** No — this one is clearly intentional (an
explicit, named validator with a clear message), just a different design
choice than decider 1 made for the same degenerate config. Not fixed
(nothing to fix; it already does what it evidently means to do).

### 4. An `InputRef` threshold cannot vary per row — only per `apply()` call

decider 1 resolves an `InputRef` as `parameters.struct.field(key)` — a
DataFrame column, so (as `test_inputref_runtime_overrides_default_per_row`
exercises directly) two rows CAN carry two different threshold values in
one call.

decider2 resolves the same `InputRef` as a kernel `param()` — a call-level
knob, explicitly documented as such in `trees/schema.py`'s divergence 3:
"Two nodes referencing the same key share one knob... A genuinely per-row
threshold is still expressible — as an ordinary input column ... but that
is not what an `InputRef` means." One `pipeline.apply(params=...)` call
sets ONE value, shared by every row that call scores. There is currently
no decider2 construction — not even the Feature-to-Feature route the
schema docstring gestures at — that reproduces decider 1's per-row
variation for an `InputRef`-bound threshold specifically (a genuine
per-row comparison would need a threshold that is itself a `Feature`, and
no unary operator's `threshold`/`min`/`max` field accepts one today).

**Is this a decider2 bug?** No — this is `trees/schema.py`'s divergence 3,
already explicitly documented as deliberate, in the source, before this
porting task began. Not fixed; not fixable without a scope decision (add a
Feature-vs-Feature comparison operator) well beyond "port the tests."
Ported as `test_inputref_cannot_vary_per_row_only_per_call`, which asserts
the real behaviour: the SAME override reaches every row of one call.

---

## Final counts

- 65 decider 1 tests examined.
- 50 ported test functions written across the 5 target files (some combine
  two decider 1 assertions that shared one fixture into a single function;
  two are new tests with no 1:1 decider 1 counterpart, added to carry a
  divergence's coverage forward — see divergences 1 and 4).
- 21 ported unchanged (same config, same answer).
- ~19 ported with mechanical adaptation (same intent/answer, different
  vocabulary or a differently-shaped supporting mechanism).
- ~9 ported as divergence-documenting tests (decider2's real, different
  behaviour, written up above).
- 15 decider 1 tests not ported at all (recorded above): 3 x
  `PrioritizedFlatRuleModule` in `test_conditions.py`, 3 x
  `PrioritizedFlatRuleModule`/`all`-mode in `test_tree_end_to_end.py`, 6 x
  decider-1-internal implementation-detail tests
  (`build_parameters_expr` x3, `prioritize_results`,
  `use_optimized_execution` x2, `RunPolarsExpression`) in
  `test_tree_end_to_end.py`, and in `test_tree_migration.py`:
  `test_v1_parses_and_upgrades_to_v3` (v1 format, KNOWN GAP) and
  `test_can_create_default_tree` (no `default_tree()` factory).
- 1 confirmed decider2 bug found and fixed (composite-node threshold
  collision, above) — the whole reason this kind of port is worth doing.
- 4 confirmed behavioural divergences found and documented, none of them
  fixed (each is either an already-documented deliberate design choice, or
  a design call outside this task's authority to make unilaterally).

Cold-cache full decider2 suite: **406 passed** (356 pre-existing + 50
ported), `rm -rf .decider2_cache .decider2_build` then `python -m pytest
tests/ -q`.
