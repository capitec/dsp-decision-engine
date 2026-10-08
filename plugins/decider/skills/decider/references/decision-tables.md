# Decision tables

Use a decision table for a grid whose rows match records: bands, rate ladders, lookup keys, limit matrices. A record takes the outputs of the first row it matches, else the `default`. Edit one with the `table_*` tools; `configs.md` has the edit loop and `try_config`.

## Document format

```json
{"type": "decision_table", "name": "rate",
 "columns": {"product": "String", "lo": "Float64", "hi": "Float64", "rate": "Float64"},
 "rows": [{"product": "loan", "lo": null,    "hi": 10000.0, "rate": 0.20},
          {"product": "loan", "lo": 10000.0, "hi": null,    "rate": 0.15},
          {"product": "card", "lo": 0.0,     "hi": 1000000.0, "rate": 0.25}],
 "expression": {"type": "and", "expressions": [
   {"type": "eq", "variable": "product", "value_column": "product"},
   {"type": "between", "variable": "amount", "lower_bound_column": "lo", "upper_bound_column": "hi"}]},
 "outputs": ["rate"], "default": [0.3]}
```

- `columns`: every column of a row, as polars dtype names: `Int64`, `Float64`, `Boolean`, `String`, `{"type": "List", "inner": "String"}`, `{"type": "Enum", "categories": ["A", "B"]}`. Dates are not supported. Every row gives a value for every column; `null` is a value.
- `expression`: how a record matches a row. `variable` names an input column of the record; the `*_column` fields name a column of the table.
- `outputs`: the columns written to the record, one output column each. A list column cannot be an output.
- `default`: one value per output, used when no row matches. Leave it out and the outputs are null.
- `rows`: the rows inline, or `{"table": "prices"}` (see Rows in params).

## Expressions

| `type` | holds when | fields |
|---|---|---|
| `eq` | the record's `variable` equals the row's `value_column` | `variable`, `value_column` |
| `between` | the `variable` lies in the row's band | `variable`, `lower_bound_column`, `upper_bound_column` (one or both), `mode`, `allow_gaps` |
| `in` | the `variable` is one of the row's list in `values_column` | `variable`, `values_column` (a `List` column) |
| `is_true` | the Boolean `variable` is true; reads no column, so it holds for every row or none | `variable` |
| `and`, `or` | every / some sub-expression holds | `expressions` |

`between` lower edge is inclusive and upper edge exclusive: `lo <= x < hi`. `"mode": "upper_inclusive"` flips that to `lo < x <= hi`. A null input never matches.

`table_view` prints the expression as one line, for example `product == row.product AND row.lo <= amount < row.hi`.

## Bands and ladders

Rows that share their `eq` values form one ladder of bands, checked on their own. In the table above the two `loan` rows are a ladder and the `card` row is another.

- A row may leave one bound `null`: it then takes the neighbouring row's edge (a missing lower is the previous row's upper, a missing upper the next row's lower).
- The first row of a ladder may leave its lower edge open (`null`), the last row its upper.
- A row with both bounds `null` in a ladder of one row cannot be resolved: give it real edges (`0.0` and a large upper) as the `card` row does.
- Unless `"allow_gaps": true`, each row's upper must equal the next row's lower. Rows are matched in order, so keep ladder rows sorted by band.

Contiguity makes an edit that moves one edge fail on its own: inserting a band with `table_add_rows` is refused with `Row 2 upper (800.0) != row 3 lower (700.0)`. Two ways round it. Make both changes in one `patch_config` (`configs.md`). Or give the table lower edges only, a `between` with just `lower_bound_column` and no `hi` column: each band then ends where the next begins, so inserting a band is one `table_add_rows`.

## Reading a table

```
table_view(path)
-> {"name": "band", "outputs": ["band"], "default": ["C"],
    "match": "row.lo <= bureau_score < row.hi  (first matching row wins; none: default)",
    "rows_in": "document", "row_count": 3,
    "rows_markdown": "| # | lo | hi | band |\n|---|---|---|---|\n| 0 | null | 600.0 | C |\n| 1 | 600.0 | 700.0 | B |\n| 2 | 700.0 | null | A |",
    "rows": [{"lo": null, "hi": 600.0, "band": "C"}, ...]}
```

Row indexes in the grid are the `index` and `indexes` the edit tools take. A large table shows `limit` rows (default 200) from `offset`; page with `table_view(path, offset=200)`.

## Editing rows

Change a cell, then check the answer:

```
table_update_rows(path, updates=[{"index": 1, "values": {"band": "B+"}}])
-> {"saved": true, "path": "..."}
try_config(path, rows=[{"bureau_score": 650}])
-> {"columns": ["bureau_score", "band"], "rows": [{"bureau_score": 650, "band": "B+"}]}
```

Add rows at an index (default: the end) and delete by index:

```
table_add_rows(path, rows=[{"product": "card", "lo": 1000000.0, "hi": 2000000.0, "rate": 0.2}], at=3)
table_delete_rows(path, indexes=[3])
```

A refused edit leaves the file as it was and says why:

```
table_update_rows(path, updates=[{"index": 1, "values": {"bnad": "B"}}])
-> {"saved": false, "problems": ["row 1 has column(s) ['bnad'] the table does not declare; ..."]}
```

Test a changed lookup on the cell it should hit and on its neighbours. For a matrix with `eq` keys, call `try_config` with the key combination you edited and one that matches nothing, which returns the `default`.

## Editing columns

Add a column to every row, optionally as a new output (the value also joins `default`):

```
table_add_column(path, name="limit", dtype="Float64", value=1000.0, output=True)
```

Then set the per-row values with `table_update_rows`. Drop a column you no longer need:

```
table_drop_column(path, name="lo")
-> {"saved": false, "problems": ["column 'lo' is used by the match expression (row.lo <= bureau_score < row.hi); change the expression first (patch_config on /expression), then drop the column"]}
```

Dropping an output column also removes it from `outputs` and `default`. Anything downstream that reads that output needs its own edit.

## Rows in params

With `"rows": {"table": "prices"}` the rows live in the params document, not in the table document. `table_view` then reports `"rows_in": "params"` and the row tools refuse. The rows are the `prices` param of the step: in `params.json` they sit at `<flow name>/<step name>/prices`, a list with one dict per row and a value for every column.

```
params_set(".../params.json", changes={"loans/price/prices": [{"product": "loan", "rate": 0.12}, {"product": "card", "rate": 0.2}]}, entry="pipeline:build")
params_set(".../params.json", changes={"loans/price/prices/0/rate": 0.1}, entry="pipeline:build")
```

With `entry`, an undeclared or misspelled column in a row is reported before the file is written. Try the table alone with its rows as params:

```
try_config(".../price.json", rows=[{"product": "card"}], params={"price": {"prices": [{"product": "card", "rate": 0.2}]}})
-> {"columns": ["product", "rate"], "rows": [{"product": "card", "rate": 0.2}]}
```

A `String` output of a params table must be an `Enum` column so its values are known up front: `"band": {"type": "Enum", "categories": ["A", "B", "C"]}`.

## Pitfalls

- Single-row edits that break contiguity are refused; combine them in `patch_config`.
- Write numbers as JSON numbers. The loader accepts `"0.5"` in a `Float64` column, but the file then keeps the string.
- `default` has one entry per output, in the same order.
- The older `"parameters": {"data": ..., "dtypes": ...}` spelling loads but the row tools refuse it; rewrite it with `columns` and `rows` first.

## Tools

- `table_view(path, offset=0, limit=200)`: columns, outputs, default, match rule and rows as a grid with indexes
- `table_add_rows(path, rows, at=None)`: insert rows at an index, default the end
- `table_update_rows(path, updates)`: change cells, `updates` is `[{"index": i, "values": {col: value}}]`
- `table_delete_rows(path, indexes)`: delete rows by index
- `table_add_column(path, name, dtype, value=None, output=False)`: add a column to every row, optionally as an output
- `table_drop_column(path, name)`: drop a column the expression does not use
- `try_config(path, rows, params=None)`: run the table on rows to check an edit
- `patch_config(path, ops, dry_run=False)`: several edits in one validated step
- `params_set(path, changes, entry=None)`: edit rows that live in the params document
