# Real trees

The tree documents the real-tree tests (`test_real_trees*.py`, `test_real_tree_imports.py`, `test_serve_exported.py`)
take through the editor: every tree document in the repo plus generated flat-rule sets. The list lives in code in
one place, `real_trees.py` (`TREES`, `EXCLUDED`); this file says why each is there and what a comparison ignores.

Each tree is a `TreeConfig` step document (`{"type": "tree", "name": ..., <step settings>, "tree": document}`). The
editor imports and exports `document` only; the step settings (`path_output`, `trace_output`, `feature_types`,
`null_handling`) live outside the editor, in the step document around it (`campaign_23.json` is one), so they are
not something an export loses: the tests put them back around the export before it runs.

## Inventory

| id | source | format | size | params | step settings |
|---|---|---|---|---|---|
| `campaign-23` | examples/campaign-trees/configs/0.1.0/campaign_23.json | v3, step document file | 18 nodes | n3_behaviour_score_floor, n6_discretionary_income_floor | `path_output`, `trace_output` |
| `mcp-flat-rules` | tests/mcp/data/flat_rules.json | prioritized, step document file | 4 rules | - | - |
| `bank-bureau-grade` | ui/vscode/examples/bank/trees/bureau_grade.json | v3, step document file | 6 nodes | excellent_threshold, fair_threshold, good_threshold | - |
| `bank-collections-risk` | ui/vscode/examples/bank/trees/collections_risk.json | v3, step document file | 4 nodes | chronic_arrears_threshold, has_judgement_threshold | - |
| `bank-fraud-screen` | ui/vscode/examples/bank/trees/fraud_screen.json | v3, step document file | 4 nodes | enquiry_burst_threshold, new_address_threshold | - |
| `stored-default-tree` | ui/tree-editor/backend/tests/data/stored_documents.json `default_tree` | v3 | 1 node | - | - |
| `stored-rich-v3` | same file, `rich_v3` | v3 | 10 nodes | codes, hi | - |
| `stored-rich-flat` | same file, `rich_flat` | prioritized | 3 rules | hi | - |
| `stored-upgraded-v1` | same file, `upgraded_v1` | v3, two subtrees | 6 nodes | - | - |
| `docs-guide-fraud-rules` | decikit/GUIDE.md | prioritized, `mode: all` | 2 rules | big_amount, new_device | `path_output` |
| `docs-guide-risk` | decikit/GUIDE.md | v3 | 2 nodes | - | `trace_output` |
| `docs-trees-loan-tree` | docs/architecture/configurable-steps/trees.md | v3 | 6 nodes | max_dti | `path_output`, `trace_output` |
| `docs-debugging-and-testing-risk-tree` | docs/guides/debugging-and-testing.md | v3 | 2 nodes | - | `path_output` |
| `docs-trees-tables-scorecards-risk` | docs/guides/trees-tables-scorecards.md (`risk_tree.json` block) | v3 | 6 nodes | max_ratio | `path_output`, `trace_output` |
| `docs-trees-tables-scorecards-risk-2` | same page | flat_rule | 1 rule | max_ratio | `path_output` |
| `docs-trees-tables-scorecards-fraud` | same page | prioritized, `mode: all` | 3 rules | big_amount | `path_output` |
| `docs-trees-tables-scorecards-fraud-2` | same page | prioritized | 3 rules | big_amount | `path_output` |
| `docs-trees-tables-scorecards-afford` | same page | flat_rule | 1 rule | min_surplus | - |
| `docs-expr-and-flowfile-output` | docs/architecture/configurable-steps/expr-and-flowfile.md (a bare tree document) | flat_rule | 1 rule | - | - |
| `campaign-scale` | examples/campaign-trees/tests/test_scale.py `random_tree(150, seed=7)` | v3, composites, no positions | 627 nodes | - | `path_output`, `trace_output` |
| `fraud-live` | examples/transaction-fraud via `fraud_interdiction/rules.py` | prioritized, `mode: all` | 521 rules | - | - |
| `fraud-shadow` | same | prioritized, `mode: all` | 114 rules | - | - |
| `fraud-overlay-base` | same | prioritized, `mode: all` | 19 rules | - | - |
| `generated-1` ... `generated-521` | `real_trees.generated(n)` | prioritized, `mode: all` | 1, 32, 64, 65, 200, 521 rules | r0_limit (not 65, 521) | `feature_types` (when a rule reads `flag`) |

How each kind is found:

- **JSON files**: every `*.json` in the repo that `TreeConfig.load` accepts as it is or wrapped
  (`test_every_tree_json_file_in_the_repo_is_in_the_list` keeps the list complete). The step document files are
  imported as their `tree` (the round trip) and as the file a user would pick
  (`test_step_document_file_imports_its_tree`).
- **Docs samples**: what the guide's and the docs' examples pass to `TreeConfig.load`: JSON blocks as written (a
  bare tree document wrapped in a step document), and Python blocks that call it, run at collection with
  `TreeConfig` and `pl` in scope (stdout silenced); a block that fails keeps what it loaded first.
- **Inline Python**: the transaction-fraud example builds its live, shadow and overlay trees in Python from its
  `rule_set` document; they are captured from `TreeConfig.load` while the example builds its pipeline. The
  campaign example's scale test builds a tree the size its analysts publish; `campaign-scale` is that tree.
- **Generated**: rule `R0` is `f0 > r0_limit` (a `parameters` block param, the editor's own form); the others cycle
  through the `fraud`, `isin_num`, `or`, `between` and `threshold` shapes of benchmarks/tree_split_sweep.py, so
  they mix `>`, `>=`, `<`, `between`, `isin`, `is_true`, a computed feature and nested `and`/`or`. The 65- and
  521-rule sets keep R0 literal: a tree with params is never split into parts, and those two sizes exist to cross
  the 64-output part boundary.

## Excluded

| document | why |
|---|---|
| stored_documents.json `default_flat` | decikit refuses to build it: an empty rule set whose output declares no columns |
| decikit/GUIDE.md `band` (two versions) | only loaded when z3 (the `prove` extra) is installed, so the e2e environment never has it; `docs-guide-risk` is the same one-node v3 shape |
| examples/transaction-fraud/configs/0.1.0/rule_set.json | a `rule_set` step type, not a tree; its trees are `fraud-*` above, and importing the file is its own test (`test_fraud_rule_set_is_refused_with_a_clear_message`) |
| docs/architecture/configurable-steps/trees.md first block, docs/architecture/lab/diff.md | they build trees from names defined outside the block (`OUT`, `leaf`, `diff`), so nothing loads |
| docs/architecture/configurable-steps/expr-and-flowfile.md `RISK` | passed to `load_flow` with a flow project the page does not include; the same two-node v3 shape as `docs-guide-risk` |

## What a comparison ignores

`canonical.py` takes out only what the editor adds or rewrites without changing what decikit runs, then compares the
rest exactly (as sorted-key JSON lines, so a failure shows the differing lines):

| normalised | why |
|---|---|
| `metadata.project_id`, `name`, `project_type`, `exported_at`, `decikit_version` | the export stamps them, and fills in `name` with the project's name when the document has none; `test_real_tree_export_keeps_the_documents_own_name` checks a document's own name on its own |
| a `flat_rule` as `prioritized_flat_rule`, `mode: first_match`, `rules: [its rule]` | the import holds one rule that way (the flat-rule editor edits rule sets); decikit runs both alike |
| an unnamed rule in a `first_match` set named `Rule <n>` | the flat-rule editor names every rule; only `all` mode names output columns after rules, where the rename is a known failure (`test_unnamed_rules_in_all_mode_keep_their_output_columns`) |
| `id` values that are UUIDs | the flat-rule editor gives every node without an id a random one. decikit names unnamed nodes by path (`"0.1"`), so a `path_output`/`trace_output` over unnamed flat-rule nodes would change; no document here relies on that |
| keys whose value is `null` | decikit reads `null` and absent alike |
| numbers as floats | the browser writes `720.0` as `720`; the param schemas (`parameters()`) are compared separately in the run test |
| `rules[].meta.position` | the flat-rule editor adds `{x: 0, y: 0}` |
| `parameters: {}`, `output.type_defs: {}` | added empty |
| v3 `type: ui-tree` / `v3-tree`, `formatVersion: 3` | the tree editor saves every tree as `ui-tree` version 3, which decikit reads as v3 |
| v3 `nodes[].position` | layout only; the tree editor lays out nodes that have none |
| v3 `edges[].data.sourceIndex` as a list | decikit reads `0` as `[0]` |
| v3 `edges[].id` | decikit never reads it; the tree editor fills in missing ones |
| `input_schema`, `subtrees` when the source has none | the editor fills them in (decikit never reads `input_schema`; one subtree naming the only root changes nothing). When the source has them, `subtrees` must come back as they were and `input_schema` must keep every feature it lists, in order, with its type (`own_parts`); the editor may add the features the tree reads that it leaves out, and every feature the editor adds must carry the type decikit reads it as |
| `parameters_col` | a field decikit never reads; the tree editor drops it |

## Running exports

- **In the library** (`test_real_trees_run.py`, `tree_runs.py`): the source and the export, each in its step document,
  load with `TreeConfig.load`, have equal `parameters()`, and run over the same probe frame with
  `parameters().defaults()`: every feature the tree reads (typed as decikit reads it), at, just below and just above
  every threshold, bound, value and pattern it compares against, with nulls (row 0 all null, then one value in ten).
  The outputs must be equal frames (columns, order, dtypes, values), or both runs fail with the same error.
- **Served** (`test_serve_exported.py`, `serve_project.py`): the export goes into a one-step project under
  `.scratch/serve-projects/served_<id>/`:

  ```
  served_<id>/__init__.py
  served_<id>/pipeline.py              def build(tree): return flow(tree, name="served")
  served_<id>/configs/0.0.1/tree.json  the export in its step document
  served_<id>/configs/0.0.1/params.json  flow(...).parameters().defaults()
  served_<id>/sample_request.json      one probe record, for the warm-up
  ```

  Every tree is scored in-process the way `decikit serve` scores it:
  `RequestHandler(JsonFileStore(basepath="configs"), "served_<id>.pipeline:build", mode="interpreted")`, `stage()`,
  `activate()`, then `process_fn(json_bytes, "application/json", "application/json")` per probe record, which must
  equal the library's row. Four trees (flat rules with params, a v3 tree, the campaign tree, the 200-rule set) also
  run `python -m decikit build` and `python -m decikit serve --port <port> --workers 1` (ports `E2E_SERVE_PORT`
  (8650) to +3) with `DECIKIT_API__CODE_PATH`, `DECIKIT_API__PIPELINE` and `DECIKIT_CONFIG__BASEPATH` set, and
  every record POSTed as raw JSON to `/invocations` must get the in-process answer. The server is stopped by its
  process when the test ends, or when the run is stopped by its PID. Only the first of those records is null:
  compiled, `run(record)` compiles again for every new set of null inputs (about 30 s each for the 200-rule set, 2.5 s
  for 32 rules; a pattern seen before costs nothing), so 50 records with scattered nulls take over 20 minutes.

## Size and time

Opening a flat-rule set renders every rule at once: 64 rules open in about 3 s, 114 in 10 s, 200 in 13 s, and 521
rules take the browser past 3 GB (the cap e2e runs have) before they open, so `fraud-live` and `generated-521` are
listed as known failures that are not run; their import, export and run without the editor are checked instead
(`test_rule_set_too_big_to_open_exports_and_runs_like_its_source`). The round trips of `fraud-shadow` and
`generated-200` take about 30 s and are marked `slow`; the 627-node `campaign-scale` tree takes about 6 s.
