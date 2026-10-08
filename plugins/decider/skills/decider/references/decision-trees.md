# Decision trees and flat rules

A tree config document (`{"type": "tree", "name": ..., "tree": {...}}`) runs one decision per row and writes the columns of its `output` table. Edit it with the `tree_*` tools. They validate the whole document before saving, so a refused edit leaves the file untouched; read `problems` in the reply and act on it.

## Pick the format

| Need | Use |
|---|---|
| A branching decision with shared sub-decisions, drawn in the VS Code tree editor | v3 tree (`nodes` + `edges`) |
| Many independent named rules, each its own yes/no (fraud rules, eligibility checks), every one reported | `prioritized_flat_rule` with `mode: "all"`; columns come out as `<rule>.<column>` |
| A waterfall where the first rule that fires answers | `prioritized_flat_rule` with `mode: "first_match"`; rule order is the priority |
| One small nested rule | `flat_rule` |
| One row per case over the same columns (rate cards, band matrices) | a decision table (`decision-tables.md`), not a tree |
| Points per variable band | a scorecard (`scorecards.md`) |

## Workflow

1. `tree_view(path)`. Read the outline and `nodes`. Node ids, branch numbers and result rows all come from here; copy them, never guess.
2. Add the output row the new leaf will select: `tree_set_output(path, {"collections_band": 4})` returns `"result_idx": 2`.
3. Add nodes top-down: `tree_add_node(path, parent, branch, data)`. The reply carries the new `id`, which is the `parent` for its own children.
4. `tree_view(path)` again to confirm the shape, then `try_config(path, rows)` with a row for each branch you touched, and compare against the answer you expect.

A worked run on a copy of `collections_risk.json`. The tree starts as `chronic_arrears` (branch 1 leads to `has_judgement`, whose branch 1 is empty and answers the default row). Goal: contactable customers with arrears over 500 and no judgement get band 4.

```
tree_set_output(path, {"collections_band": 4})
  -> {"saved": true, "result_idx": 2, "added_dtypes": []}
tree_add_node(path, "has_judgement", 1, {"type": "composite", "op": "and", "conditions": [
    {"op": ">", "feature": "arrears_amount", "threshold": {"param": "arrears_floor", "default": 500}},
    {"op": "is_true", "feature": "contactable"}]})
  -> {"saved": true, "id": "and_of_2"}
tree_add_node(path, "and_of_2", 0, {"type": "leaf", "result_idx": 2})
  -> {"saved": true, "id": "leaf_2"}
```

`tree_view` now ends with:

```
    [1 false] and_of_2: (arrears_amount > {arrears_floor=500} AND contactable is true)
      [0 true] leaf_2: leaf -> row 2 {"collections_band": 4}
      [1 false] -> default row {"collections_band": 1}
```

`try_config(path, rows=[{"missed_payments_12m": 0.0, "judgements": 0.0, "arrears_amount": 900.0, "contactable": true}, {"missed_payments_12m": 0.0, "judgements": 0.0, "arrears_amount": 900.0, "contactable": false}])` gives `collections_band` 4 then 1.

Other edits:

- `tree_set_node(path, node_id, data)` replaces a node's data; children stay on their branch numbers. Refused while children sit on branches the new type lacks (a leaf has none): remove them first.
- `tree_remove_node(path, node_id)` removes the node, its incoming edges and every descendant nothing else leads to. The reply lists `removed`, `kept_shared` (descendants with another parent) and `now_default` (branches that now answer the default row).
- To replace a child: `tree_remove_node` it, then `tree_add_node`. Adding to an occupied branch is refused, naming the occupant.
- `tree_set_output(path, values, index=1)` replaces row 1; `index=-1` sets the default row. A new column needs `dtypes={"reason": "String"}`; the refusal suggests one.
- Flat rules: `tree_add_rule(path, {"meta": {"name": "vip"}, "rule": {...}}, at=0)` inserts at a position (default the end); `tree_remove_rule(path, "vip")`. Names are unique.

## Document format

v3 tree: `tree` is `{"nodes": [...], "edges": [...], "output": {...}}`. A node is `{"id": "root", "data": {...}}`; an edge is `{"source": "root", "target": "high", "data": {"sourceIndex": 0}}`. The root is the node no edge points at. `sourceIndex` may be a list (`[1, 2]`): one edge wiring one child to several branches. A branch with no edge answers the default row. A node may have several parents; it is shown once in the view and referenced after.

| Node `data` | Branches |
|---|---|
| `{"type": "leaf", "result_idx": 0}` | none; selects `output.data[0]`, and `-1` selects `output.default` |
| `{"type": "unary", "condition": {...}}` | 0 = condition holds, 1 = it does not |
| `{"type": "composite", "op": "and"\|"or"\|"not", "conditions": [...]}` | 0 = holds, 1 = does not; `not` takes exactly one condition; conditions may nest composites |
| `{"type": "cases", "op": "ranges", "feature": "score", "conditions": [{"max": 600}, {"min": 600, "max": 700}, {"min": 700}]}` | branch i = band i, last branch = otherwise. Bands are `[min, max)`; `"end_logic": "upper_inclusive"` gives `(min, max]`. Strict (default): bands sorted and contiguous |
| `{"type": "cases", "op": "isin", "feature": "n", "conditions": [{"values": [1, 2]}, {"values": [3]}]}` | branch i = first case holding the value, last = otherwise |
| `{"type": "cases", "op": "string_match", "feature": "s", "conditions": [{"patterns": ["web"]}], "match_type": "starts_with"}` | same; `case_sensitive`, `trim_whitespace` apply to all cases |

Conditions (unary `condition`, composite members) each have `feature` plus:

| `op` | Fields | Reads as |
|---|---|---|
| `>` `>=` `<` `<=` `==` `!=` | `threshold` | `ratio > 0.7` |
| `between` | `min`, `max` (either may be left out; closed) | `a between 1 and 5` |
| `isin` | `values`: list of numbers, or `{"param": ...}` | `n in [1, 2]` (numbers only; use `string_match` for text) |
| `string_match` | `patterns`, `match_type` exact\|starts_with\|contains\|ends_with\|regex, `case_sensitive`, `trim_whitespace`, `null_handling` no_match\|match\|error | `s starts_with ["web"]` |
| `is_true` `is_false` | none | `flag is true` |

- `feature` is a column name, or a computed feature `{"type": "computed", "expression": "income - debt"}` over columns. A threshold that differs per row is a computed feature compared with `0`.
- Any number or pattern may be `{"param": "hi_thresh", "default": 0.7}`: a param of the step, retuned per run without editing the tree (`params_set`). Param names are Python identifiers.
- `output` is `{"data": [{"g": "C"}, ...], "default": {"g": "B"}, "dtypes": [["g", "String"]]}`. A leaf's `result_idx` indexes `data`; a column a row omits is null; every column in `data` needs a `dtypes` entry.

Flat rules: `prioritized_flat_rule` has `rules: [{"meta": {"name": ...}, "rule": <node>}]` and `mode`; `flat_rule` has one `rule`. A rule node nests its branches: unary and composite take `then` (branch 0) and `otherwise` (branch 1), either left out for the default row; a cases node is `{"type": "cases", "op": "isin", "feature": ..., "conditions": [{"when": {"values": [1, 2]}, "then": 0}], "otherwise": 2, "branches": [leaf, leaf, leaf]}`, where `then` and `otherwise` index into `branches`. Nodes without an `id` get positional ids (`0`, `0.0`, `1.2`); `tree_view` shows them, and `node=` takes a rule name or an id.

Step-level fields, beside `tree`:

- `null_handling`: `"otherwise"` (default; a null feature fails its test and takes branch 1, a cases node tries the next case) or `"error"`.
- `feature_types`: `{"n": "int64"}` to compare an integer column exactly; otherwise features are inferred (`string_match` text, `is_true` bool, else float).
- `path_output: "leaf"` adds a String column with the id of the answering leaf (null when the default row answered); `trace_output: "path"` adds the ids walked, joined by `>` (`band>channel>netflag>hi`). In `mode: "all"` each is per rule.

## Pitfalls

- `first_match` order is priority; `at=` decides it. Inserting a rule shifts the positional ids of later unnamed nodes, and with them `path_output` values and `<name>#<id>` locators. Give nodes explicit `id`s where something reads them.
- Add the output row before a leaf that selects it; `tree_add_node` refuses a `result_idx` past the end.
- The tools cover structure and rows. For one field inside a node (a threshold), `tree_set_node` with the whole new `data`, or `patch_config`.
- A second root cannot be added to a non-empty tree; a node without a path from the root never runs and is listed under `unreachable`.
- A cases node's last branch is always otherwise, so changing its case count moves that branch; the reply `note` says so. Re-run `tree_view`.
- `tree_view` of a 500-rule document is long: pass `node="<rule name>"`.

## Tools

- `tree_view(path, node=None)`: outline plus `nodes` (id, type, condition, children by branch, parents), `unreachable`, `rules`; `node` is a node id or rule name
- `tree_add_node(path, parent, branch, data, id=None)`: add a v3 node under `parent` at `branch`; `parent=None` adds the root of an empty tree
- `tree_set_node(path, node_id, data)`: replace a v3 node's data
- `tree_remove_node(path, node_id)`: remove a v3 node and its private descendants
- `tree_set_output(path, values, index=None, dtypes=None)`: append (returns `result_idx`) or replace an output row; `index=-1` is the default row
- `tree_add_rule(path, rule, at=None)`: insert a flat rule into a `prioritized_flat_rule` document
- `tree_remove_rule(path, name)`: remove a flat rule by `meta.name`
- `try_config(path, rows, params=None)`: run the document's step on rows and return the output rows
- `patch_config(path, ops, dry_run=False)`: JSON Patch for edits the tree tools do not cover
- `params_set(path, changes, entry=None)`: set values in the params document, which retunes every `{"param": ...}` value without touching the tree
