# Gap coverage and technical debt

Status of the items in `notes/tmcfeval/DECIDER_GAPS.md` and the deferred work
from the `each` / `Columnar` work. This is a living ledger: tick an item here
when it lands, and keep its detail in its own note.

## Shipped since `DECIDER_GAPS.md`

- 3. Lists of records have no grain of their own — `each` + `Columnar[Item]`.
- 4. No fan-out and reduce — `optimise`. With 3 and 4 landed, gap 5
  ("calling a flow from a hand-written kernel") is withdrawn as written.
- 6. A branch condition must be a function step — `branch("on_card", ...)`.
- 2. Silent Python fallback when a step calls a plain helper — `@helper` /
  `@allow_fallback`, with a `FallbackWarning` (or raise in `strict_compile`).
- 8. Functions with function arguments can't be disk-cached — moot: `optimise`
  owns the search loop now, so decider's kernels never take functions as args.
- 13. Design note: should helper calls be traced? — closed by decision (not
  recommended; business meaning is a step, helpers are arithmetic inside one).
- 9. Frame step output dtypes are lost between frame steps — `frame_step`
  now takes a `writes` dict declaring each output's polars dtype
  (`writes={"clean": pl.List(pl.Struct(...))}`), and `State.frame_of` /
  `State.column` reuse the Series a frame step returned, so even an
  *undeclared* all-empty nested column keeps the dtype it was written with,
  both between frame steps and in the final output.

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

## Still open in `DECIDER_GAPS.md`

- 10. Typo heuristic flags legitimate input names.
- 11. Per-record overhead.
- 12. Template nests the package (and a place for extensions).

## Deferred (not in `DECIDER_GAPS.md`)

- Type handling on a step's declared columns. Gap 9 now *preserves* dtypes;
  it does not yet *verify or cast* a frame step's declared `writes` against
  what it returns. A `type_mode`/`TYPE_HANDLING`-style knob (verify vs cast,
  warn vs raise) and polars-typed `reads` were discussed but deliberately left
  out: `reads` stays Python-typed because it also drives JSON date coercion
  (`notes/tmcfeval/DECIDER_GAPS.md` gap 9, user feedback). Applying the same
  idea to scalar `@step` inputs is a further follow-up.
