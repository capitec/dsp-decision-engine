# Can `Columnar[Item]` read Arrow directly and join the shared kernel?

Date: 2026-09-26 · Script: `benchmarks/nested_rows.py`, `benchmarks/bundle_search.py`

## Purpose
Two questions on a list-of-records column: can its field arrays be views over Arrow buffers, and can a step reading it join the shared array kernel (so a `loop` over it packs) instead of one dispatcher call per row?

## Method
`build_rows` at 1 to 200k rows (3 items, 2 fields) with the Arrow read forced on and off; a `list[dict]` step as the bar (20k rows, light and heavy body). Then a `loop` over bundle masks reading the items, before and after slicing the flat arrays inside the kernel, against a hand-written `@njit` search. Loaded shared box, `fused`: read ratios, not absolutes.

## Result
`build_rows`, microseconds:

| rows | copy loop (before) | Python values | Arrow |
|---|---|---|---|
| 1 | 19 | **9** | 130 |
| 256 | 996 | 598 | **570** |
| 200,000 | 763,575 | 457,653 | **343,381** |

- Arrow has a fixed ~130 us cost, so it wins from 256 rows (2.2x the old loop at 200k). A single record went 19 to 9 us with no Arrow: per-plan schema cache, `np.fromiter`, no slicing for one row.
- Against `list[dict]`, per-row dispatch lost with a light body (0.4-0.6x on `score()`, a fixed ~23 us) and won with a heavy one (up to 4.2x at 400 items). The annotation is a compute optimisation.
- In the kernel, `score()` for 9 items x 511 masks fell 60.6 ms to 0.10 ms; `run()` over 100 orders 576 to 3.5 ms; at 16 items x 65,535 masks 7,174 ms to 10.4 ms, 1.07x of hand-written. The loop packs (`['order/search']`, no fallbacks).
- Single record also improves (two `Columnar` steps, 3 items: 97 to 42 us); no path regressed reproducibly.
- Trap: Arrow-backed arrays are `readonly array(...)`, a distinct numba type; type the namedtuple from the arrays, not a writable probe. `Columnar[Item] | None` cannot join (no in-band `None`).

## Decision
Arrow read from 256 rows, Python values below, `score()` always Python; the shared-kernel join is the only behaviour. See [nested data](../decisions/nested-data.md).
