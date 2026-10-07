# Decoding `Literal` outputs

A compiled step declared `Literal[...]` (a tree's string outputs, `path`, `trace`) returns each
row's index into the choices. The kernel keeps it as a `Coded` (codes plus a `LiteralTable`
built once per kernel) and decodes it only where it is read:

- `score()` and every Python reader (`State.read`) index the cached object array: about 0.1 µs
  for one row, and no polars.
- `run()` (`State.column`) gathers from the cached polars `String` column.
- A partial-row write (inside a branch arm) decodes to objects first.

The alternative was one decode for both paths. An object column turned into polars costs
2 µs at one row but grows with the row count (23 µs at 1k, 2 ms at 100k), while
`Series.gather` costs a flat ~13 µs up to a few hundred rows, then about a tenth as much per row.
`score` is the single-record path the serving API uses, so it gets the object lookup, and `run`
gets the gather.

Before this, the choices array was rebuilt on every call (O(leaves)), and `score()` hashed each
output's annotation per call; hashing a `Literal` of thousands of leaves walks all its args.

Fused mode, one String output unique per leaf (`score` µs / `run` µs at n = 1, 1k, 100k):

| leaves | before | after |
|---|---|---|
| 64 | 17 / 53, 213, 16 100 | 13 / 64, 114, 5 900 |
| 8192 | 639 / 692, 951, 25 000 | 14 / 74, 199, 14 600 |

The extra ~11 µs on a one-row `run()` is gather's fixed cost.

Struct and list outputs constant per leaf are meant to reuse this: one table per output,
gathered by `run`, indexed by `score`.
