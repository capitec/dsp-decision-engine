# Scorecards

Use a scorecard to turn inputs into points and sum them: each variable's value falls into a bin that carries points, and the card writes every variable's points plus the total. Edit one with `scorecard_view` and `scorecard_set_bins`; `configs.md` has the edit loop and `try_config`.

## Document format

```json
{"type": "scorecard", "name": "card", "output_name": "total", "variables": [
  {"type": "constant", "score": 600, "output_name": "base"},
  {"type": "scored", "variable_name": "age", "default": {"value": 0, "name": "unknown"},
   "bins": [{"value": -20, "upper_bound": 25, "name": "young"},
            {"value": 10, "lower_bound": 25, "upper_bound": {"param": "old", "default": 60}},
            {"value": 0, "lower_bound": {"param": "old", "default": 60}}]},
  {"type": "scored", "variable_name": "tier", "default": {"value": 0},
   "bins": [{"value": 15, "items": ["gold", "platinum"]}, {"value": 5, "items": ["silver"]}]},
  {"type": "adjusted", "scale": 2.0, "offset": -5.0,
   "variable": {"type": "scored", "variable_name": "util", "default": {"value": 0},
                "bins": [{"value": 10, "upper_bound": 0.3}, {"value": 0, "lower_bound": 0.3}]}}]}
```

- `name`, `output_name` (default `score`): the total, the sum of every variable's output.
- `variables` (at least one, `variable_name`s unique):
  - `scored`: bins `variable_name` into points, written to `{variable_name}_score` (`value_output_name` renames it).
  - `adjusted`: wraps a `scored` variable and writes `points * scale + offset` to `{variable_name}_adjusted_score`; the total adds the adjusted points, and the plain points are written too.
  - `constant`: the same `score` for every row, written to `output_name`.
- Every variable's points are output columns, so a decision record shows which variable contributed what.

## Bins

A `scored` variable has `bins`, a `default`, and `strict` (default `true`).

- A bound bin has `value` (points), optional `lower_bound`, `upper_bound`, optional `name`. It matches `lower_bound < x <= upper_bound`: lower exclusive, upper inclusive. A missing bound is the neighbouring bin's edge, or open at the ends of the list.
- A values bin has `value` and `items`, a list of numbers or strings; it matches when the input equals one of them. Values bins are checked before bound bins, and an item may appear in only one bin.
- `default` (`value`, optional `name`) scores when no bin matches or the input is null.
- Bound bins must be in order and not overlap; with `strict` each bin's lower bound must equal the previous bin's upper bound. `"strict": false` allows gaps, which score the `default`.
- A number can be a param: `{"param": "old", "default": 60}` makes `old` a param of the scorecard, and it applies to `value`, bounds, `scale`, `offset` and `score`. Retuning it through the params document needs no rebuild.

The param `old` of variable `age` in the card above lives in `params.json` at `<flow name>/card/age/old` (and at `card/age/old` when the card runs alone, as in `try_config`).

## Reading a card

```
scorecard_view(path)
-> {"name": "card", "output": "total",
    "variables": [
      {"kind": "constant", "output": "base", "points": "600"},
      {"variable": "age", "kind": "scored", "output": "age_score", "strict": true,
       "bins_markdown": "| # | matches | points | name |\n|---|---|---|---|\n| 0 | x <= 25 | -20 | young |\n| 1 | x > 25 and x <= {old=60} | 10 |  |\n| 2 | x > {old=60} | 0 |  |\n| default | no bin matches, or null | 0 | unknown |",
       "bins": [...], "default": {...}},
      ...],
    "params": {"card/age": {"old": {"type": "float", "default": 60}}}}
```

`{old=60}` is a param with its default. `params` lists every param with its path, so you know what `params_set` can retune.

## Replacing bins

`scorecard_set_bins(path, variable, bins)` replaces one variable's whole bin list, so send every bin, not only the changed one.

```
scorecard_set_bins(path, variable="age", bins=[
  {"value": -20, "upper_bound": 25, "name": "young"},
  {"value": 10, "lower_bound": 25, "upper_bound": 50},
  {"value": 0, "lower_bound": 50}])
-> {"saved": true, "path": "..."}
```

A list that breaks the ordering rules is refused and the file is untouched:

```
scorecard_set_bins(path, variable="age", bins=[{"value": -20, "upper_bound": 30}, {"value": 10, "lower_bound": 25}])
-> {"saved": false, "problems": ["variables.1.scored: BoundBin at index 1 has lower_bound 25.0 which is less than previous highest_bound 30.0. Bound bins must be in order and non-overlapping."]}
```

Then check with rows on both sides of every edge you moved. A row at the edge itself belongs to the lower bin:

```
try_config(path, rows=[{"age": 25.0, "tier": "gold", "util": 0.1}, {"age": 26.0, "tier": "gold", "util": 0.1}])
```

The reply carries `age_score` (-20 then 10), `tier_score`, `util_score`, `util_adjusted_score`, and `total`.

## Other edits

Change one points value, add a variable, or retune an edge with `patch_config`; a card is an ordinary document.

```
patch_config(path, ops=[{"op": "replace", "path": "/variables/1/bins/0/value", "value": -30}])
patch_config(path, ops=[{"op": "add", "path": "/variables/-", "value": {"type": "constant", "score": 50, "output_name": "bonus"}}])
```

To move an edge that is a param, change the params document: `params_set(".../params.json", changes={"credit/card/age/old": 55}, entry="pipeline:build")`. `try_config` takes the same change as `params={"card": {"age": {"old": 55.0}}}`.

## Pitfalls

- A bound of exactly `0` is read as unset, so `{"value": 1, "lower_bound": 0}` also matches negative inputs. For a split at zero use a small nonzero edge (`0.0001`), and check with a negative row.
- `scorecard_set_bins` replaces the list; leaving a bin out deletes it.
- A null input scores `default`, not zero. Set `default` to the points you intend, and test a null row.
- The input type follows the `items`: a variable with string items reads a string input and one with numbers reads a number, so a row of the other type fails with a type error instead of scoring the default.

## Tools

- `scorecard_view(path)`: variables, bins as tables, defaults and the params the card declares
- `scorecard_set_bins(path, variable, bins)`: replace one variable's bins
- `try_config(path, rows, params=None)`: score rows to check an edit
- `patch_config(path, ops, dry_run=False)`: edits outside the bins
- `params_set(path, changes, entry=None)`: retune a param edge or value
