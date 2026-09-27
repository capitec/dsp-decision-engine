# Gap coverage and technical debt

Status of the items in `notes/tmcfeval/DECIDER_GAPS.md` and the deferred work
from the `each` / `Columnar` work. This is a living ledger: tick an item here
when it lands, and keep its detail in its own note.

## Each: what shipped, what did not

`each(column, child_flow, name=, execution_mode=EachMode.PER_ROW | BATCH)` ships
with both modes, child params tunable through the parent's params document, and
null/empty lists reading as no items. Two things were deliberately left out.

### `break_at` / sessions into the child grain

`pipeline.session(df).break_at("order/items/heavy")` cannot step into the child
flow's nodes. The child plan is bound as its own `Executable` inside the `each`
node, not embedded in the parent's plan, so the session's checkpoint walk never
sees its nodes. To support it, the child plan's nodes need to be reachable from
the parent's `iter_nodes`/session machinery (a new IR node carrying the child,
or an equivalent), and `Checkpoint` needs a coordinate for *which item* — see
`notes/nested-data-decisions.md` ("`Checkpoint` ... has no coordinate for which
item").

### Edge cases deferred to the next change

- **`_concrete` Null→String** (`engine.py:263-271`): an all-null list column is
  rewritten to `String` inside nested types, so `each(..., BATCH)` on a single
  record with an empty or null list fails the `explode`/`unnest` (the child
  frame's field reads as a string). Fix the rewrite to stop at nested `Null`
  fields. Worse for a batch shape: a whole column, not one row, can be all-null.
- **`from_series` / `build_rows` re-conversion** (`rows.py`, `state.py`): the
  enriched list a `BATCH` `each` writes is a frame-step output, never traced to
  the input's Arrow buffers, so a later `Columnar[Item]` read pays
  `Series.to_list()` + a Python loop per item instead of the `ARROW_ROWS` fast
  path.

## Still open in `DECIDER_GAPS.md`

- 2. Silent Python fallback when a step calls a plain helper.
- 4. No fan-out and reduce (search over generated candidates) — `optimise`.
- 6. A branch condition must be a function step.
- 8. Functions with function arguments can't be disk-cached.
- 9. Frame step output dtypes are lost between frame steps.
- 10. Typo heuristic flags legitimate input names.
- 11. Per-record overhead.
- 12. Template nests the package (and a place for extensions).
- 13. Design note: should helper calls be traced?
