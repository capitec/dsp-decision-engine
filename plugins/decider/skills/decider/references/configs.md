# Config documents and params

Read this before editing any JSON a decider project keeps outside Python: trees, decision tables, scorecards, custom step documents, and the params document. Table and scorecard formats are in `decision-tables.md` and `scorecards.md`; trees in `decision-trees.md`.

## Where documents live

A project keeps its documents in `configs/<version>/`. `def build(band)` in `pipeline.py` receives `configs/<version>/band.json`, loaded, as its `band` argument; a nested file `risk/grade.json` arrives under the key `risk.grade`. `decider build` and `decider serve` use the highest version.

`configs/<version>/params.json` is the params document: tuned values, keyed by step path.

```json
{"credit": {"approved": {"limit": 0.4}}}
```

`credit` is the flow's name, `approved` the step, `limit` a `param()` of that step. A top-level `shared` key holds params several steps read. Any number inside a tree, table or scorecard can also be `{"param": "young", "default": 25}`, which adds a param named `young` under that step's path.

Treat a version that has served traffic as immutable: stored decisions replay against it. To change a live project, copy the version directory to the next number (`cp -r configs/0.1.0 configs/0.2.0`) and edit the copy. Documents outside `configs/` (a `trees/` folder, say) are edited in place.

## Edit loop

1. `list_configs(root)` finds every document: path, `kind` (the step `type`, or `params`), `name`, `version`, `key`. Use the returned absolute paths in later calls.
2. `read_config(path)` returns the document and `problems`. Fix `problems` before anything else.
3. Edit with the structured tool for the kind: `table_*`, `scorecard_set_bins`, `tree_*`, `params_set`. Reach for `patch_config` for anything those do not cover.
4. Read the reply. `saved: true` means the document built and was written. `saved: false` with `problems` means the file is untouched; read the problem, correct the edit, send it again.
5. Run `try_config` on rows that pin down the intended answer: one row per changed branch, rows on each side of a boundary you moved, a null input. Compare against the answer you expected before the edit.
6. When the project has a pipeline, `params_view(path, entry)` against its entry confirms the params document still fits.

Every write goes through the same check: the document is built as a step, and a document that fails is never saved. The file's own JSON layout (indent, trailing newline) is kept, so a one-value edit is a one-line diff.

## patch_config

`ops` is an RFC 6902 JSON Patch: `add`, `remove`, `replace`, `move`, `copy`, `test`. Pointers use `/` between keys, `~1` for a `/` and `~0` for a `~` inside a key, and `-` to append to an array. Ops apply in order, and the document is checked once, after the last op, so one call can make an edit that no single op could.

Splitting the top band of a ladder: moving its lower edge alone leaves a gap, so both ops go together.

```
patch_config(path, ops=[
  {"op": "replace", "path": "/rows/2/lo", "value": 800.0},
  {"op": "add", "path": "/rows/2", "value": {"lo": 700.0, "hi": 800.0, "band": "A-"}}])
-> {"saved": true, "path": "..."}
```

A failing op names its index and pointer, and nothing is written:

```
patch_config(path, ops=[{"op": "replace", "path": "/rows/9/hi", "value": 650.0}])
-> {"saved": false, "problems": ["op 0 ('replace' at '/rows/9/hi'): index 9 is out of range for an array of 3"]}
```

Put a `test` op first when the edit assumes a current value: `{"op": "test", "path": "/name", "value": "band"}` stops the patch if the document is not the one you read.

`dry_run=true` returns the patched `doc` and its `problems` without saving, for checking an edit you are unsure of.

## try_config

`try_config(path, rows, params=None)` runs the document's step on `rows` (a list of dicts of input columns) and returns `columns` and `rows`: each input plus the step's outputs. A document that does not build returns `problems` instead.

```
try_config(".../band.json", rows=[{"bureau_score": 650}, {"bureau_score": 900}])
-> {"columns": ["bureau_score", "band"], "rows": [{"bureau_score": 650, "band": "B"}, {"bureau_score": 900, "band": "A"}]}
```

- Run on the document alone, the step's params are keyed by its `name`: `params={"price": {"prices": [...]}}` for a table named `price` with `"rows": {"table": "prices"}`. Inside a flow they are keyed by flow name and step name, as in `params.json`.
- A null input never matches a table row, so include one when the answer for missing data matters.
- Rows enter a polars frame as given; check date inputs through the pipeline.

## Params documents

`params_view(path, entry=None)` returns the document. `params_set(path, changes, entry=None)` sets values by slash path, creating missing levels:

```
params_set(".../params.json", changes={"credit/approved/limit": 0.3})
-> {"saved": true, "path": "..."}
```

A list element has its index as a path part: `{"loans/price/prices/0/rate": 0.1}`. To replace a whole table, set the table's path to the new list of rows.

Give `entry` to check against the pipeline: the same string as `DECIDER_API__PIPELINE` in the project's `.env` (`pkg.pipeline:build`, or `pipeline:build` when `pipeline.py` sits in the project root), a `module:flow_object`, or a flow-file path relative to the project. With `entry`, `params_view` lists `problems` and returns the `schema` of every valid path. `params_set` refuses a change that adds a problem, and names the closest valid key:

```
params_set(".../params.json", changes={"credit/approved/limt": 0.3}, entry="pipeline:build")
-> {"saved": false, "problems": ["credit/approved: unknown param 'limt'; did you mean 'limit'?"]}
```

A wrong step path gets the same treatment (`no step with params at 'loans/pricee'. Did you mean 'loans/price'?`). Problems that were already in the document stay in `problems` on a saved reply and do not block the edit.

## Pitfalls

- `write_config` refuses an existing path unless `overwrite=true`; use it to create a new version's document, `patch_config` to change one.
- `params.json` is checked as a JSON object only; its values are checked when you pass `entry`.
- An `entry` that builds needs its imports to work (sibling projects on `PYTHONPATH`, as `decider build` needs). The pipeline source is re-imported on each call, so edits to it are seen.
- A document edit never recompiles anything; run `decider build` before shipping.

## Tools

- `list_configs(root=".")`: every config document under `root` with kind, name, version and key
- `read_config(path)`: the document and its `problems`
- `validate_config(path=None, doc=None)`: `problems` for a file or an unsaved document
- `write_config(path, doc, overwrite=False)`: create a validated document, making directories as needed
- `patch_config(path, ops, dry_run=False)`: RFC 6902 patch, validated and saved
- `try_config(path, rows, params=None)`: run the document's step on rows and return the outputs
- `params_view(path, entry=None)`: the params document, checked against a pipeline when `entry` is given
- `params_set(path, changes, entry=None)`: set values by slash path
