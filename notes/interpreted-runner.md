# The interpreted runner and run-time params

`engine/run/` choices that the session (T1.5) and the compiled runners (T3.3)
build on.

- **State stores raw columns plus masks; null policy is applied per reader.**
  Two steps may read one column with different policies (`x: float` and
  `x: float | None`), so input columns are loaded once as nullable (values +
  validity mask) and each call applies its own `Input.null_policy`. Float, int
  and bool inputs are loaded through `boundary.extract` (same casts as the
  compiled path); strings and unannotated columns stay Python objects, so an
  interpreted step compares a `str` against a `str` param directly.
- **Branches and loops run on row subsets.** A scope carries the absolute row
  indices it runs on; every write scatters into full-length arrays and marks
  the untouched rows invalid, so an arm's version emitted with `name@path` is
  null on rows that took another arm. An arm no row takes is skipped entirely:
  no checkpoints, no params validation, which is what lets lazy validation
  ignore it.
- **Loops** evaluate the condition on the rows still looping; rows reaching
  `max_iterations` stop without an error.
- **Frame steps** get the scope's frame (the input frame, or the last
  unknown-lineage frame's output) with every value computed so far overlaid,
  and must return the same number of rows in the same order; filtering belongs
  outside the pipeline.
- **Checkpoints:** every node reached yields `before` and `after`; none per
  row. A session breaks on `before` and inspects on `after`.
- **Params:** interpreted mode checks a node's status right before calling it,
  so it never needs the defaults bundle; the compiled runners do. Eager mode
  validates every node with params the first time a document is seen (and
  again while it stays invalid). Every key in the document must name a path
  prefix of a node with params (or a declared `shared` key), checked at every
  depth with a did-you-mean.
