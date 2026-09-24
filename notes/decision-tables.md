# Decision tables: one matcher over packed rows, rows inline or from params

`decider.steps.tables` choices, built on the tree pattern (`tree-walker.md`).

- **One row node per table.** `DecisionTableConfig.to_ir` emits one `row`
  `CallNode` whose `fn` is `matcher.match`, one njit function for every
  table. The conditions are a jump program (`(kind, variable slot,
  condition, lower op, upper op, then, else)` rows, AND/OR as jumps, no
  DNF), passed by address like a tree's. The rows are a `Rows` value: the
  addresses of one int64, one float64 and one byte array plus the row
  count. The ints start with a header, two starts per condition and per
  output, because where each part begins depends on the rows. `between`
  bounds are resolved (decider_old's neighbour fill) when rows are packed,
  so the kernel reads a finished `(lo, hi, has_lo, has_hi)` per row.
- **Rows inline or from params, one kernel.** Inline rows are packed at
  `to_ir` and sit last in the node's consts; `{"table": "rows"}` makes a
  table param, and the rows arrive in the params bundle. Both are the same
  `Rows` type (a namedtuple subclass, typed by numba as a named tuple of
  four int64), so an overload picks the source at typing time and one
  compiled matcher serves both. Changing the rows or their count never
  changes a type, so it never recompiles (`no_recompile` tests cover run,
  score and a zero-row table).
- **Rows are checked and packed when the document arrives.** The table
  param is `ctx.table(...)`'s `ParamDecl` with an `AfterValidator` in its
  `field_info`: validating the params document casts the rows to the
  declared polars dtypes (an unknown or missing column, a value that
  doesn't fit, a String outside an Enum's categories, a gap between bands
  are `ParamsError`s) and returns the packed `Rows`. The params cache keeps
  the validated bundle, so a document is packed once and the arrays live as
  long as its cache entry. `field_info` is left out of `ParamDecl`
  equality, so two tables can share one shared table param.
- **String outputs of param rows must be Enum columns.** A kernel returns a
  `Literal` output as the index of its value, and the choices are part of
  the IR. Inline rows give them; rows that arrive later don't, so such a
  column is declared `{"type": "Enum", "categories": [...]}` and adding a
  category is a schema change, which rebuilds the IR as a schema change
  should. The alternative, running the table in Python whenever it has a
  String output, would have made the common case slow.
- **Output types follow the rows.** An inline table's output is `T | None`
  only if the default or some row's value is null, as trees do; param rows
  can hold anything, so their outputs are always `T | None`. So an inline
  edit that introduces the first null changes the output type and
  recompiles once.
- **Semantics are decider_old's.** First matching row wins, else `default`
  (nulls when there is none). decider_old evaluated a table as polars
  `when/then`, so a comparison with a null is null and never matches; that
  covers a null input, a null `eq` value in a row (decider2 treated that as
  a wildcard; decider_old doesn't) and a null list. There is no NOT in the
  table vocabulary, so null and false never need telling apart.
  `in` also takes numeric lists (decider_old cast every list to strings);
  int columns are compared as float64. `unnest_output` is accepted and
  ignored: outputs are always flat columns.
- **Strings are matched raw.** `eq` on a String column and `in` on a
  List[String] compare the input's UTF-8 span with the row's bytes in the
  kernel, through the tree walker's `_matches`; nothing becomes a code.
- **The reference is independent.** It walks the expression model over the
  typed rows as dicts, resolves bands itself and calls `visit(str(r))` per
  row tried, so `<name>#<r>` is a session locator. `test_corpus.py` checks
  it against the kernel on random tables and records with nulls.
- **Null inputs.** A number variable is `missing_as(NaN)`, which no
  comparison matches (so a NaN in the data never matches either); bool and
  string variables are `T | None` (a null span has length -1).
- **Engine edits.** A node with a table param skips the compile probe
  (`_probe_signature`: the defaults bundle has no `Rows`), so it compiles
  on first use; a required param whose type can't be built empty no
  longer breaks the defaults bundle (`_placeholder` catches `ValueError`);
  the probe types an OPTIONAL `T | None` input as `Optional(T)`, as the
  runners pass it (it used float64, so a `bool | None` input compiled a
  useless float specialisation first, and a table's failed as a fallback);
  a kernel launch shares one all-valid mask among its OPTIONAL inputs.

**Measured** (`benchmarks/tables_vs_decider2.py`; 12 bands writing String,
Int64 and Float64 outputs; 1M rows through `run(df)`, 20k `score(dict)`;
dev box at load average ~4, 2026-09-23; an earlier run at load ~14 gave
the same order):

| table | engine | ns/row | score p50 | score p99 |
|---|---|---|---|---|
| bands | decider2 fused | 304 | 42.9 µs | 53.1 µs |
| bands | decider fused | 164 | 27.4 µs | 36.1 µs |
| bands | decider fused, rows param | 178 | 95.9 µs | 109.2 µs |
| bands AND string `in` | decider2 fused | 398 | 75.4 µs | 86.8 µs |
| bands AND string `in` | decider fused | 413 | 69.6 µs | 111.6 µs |
| bands AND string `in` | decider fused, rows param | 423 | 137.1 µs | 157.4 µs |

The rows-param rows pay for `document_key`, which JSON-encodes the whole
params document on every call (about 40 µs for 12 rows, profiled). Now
`ParamsCache.key` remembers the last document object and its hash, so a
document passed again is not re-hashed: rows-param `score()` p50/p99 went
from 88.6/191.9 µs to 35.8/42.9 µs (dev box, load ~5). The cache holds the
document, so its id can't be reused while it's cached; documents are
treated as immutable once passed (`run`/`score` docstrings say so). A
path that alternates documents on one executable still hashes each call;
widen the one slot to a small dict if that shows up.

Decision tables have no matched-row output (the tree's `path_output`
counterpart): it needs a new output kind in the matcher's `pick` and the
reference, so it waits for someone to ask.
