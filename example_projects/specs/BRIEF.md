# Implementer brief (read fully before starting)

You are implementing one of the example projects in `example_projects/specs/`
with the `decider` framework. Several agents build the same projects
independently; your work will be compared, served, summarised and judged, so
follow this brief exactly.

## What to read

1. Your project's spec (`example_projects/specs/<NN>-*.md`) and its slice in
   `SCOPE.md`. Build the slice; the rest of the spec is context.
2. `README.md` (the slate), `DEPS.md` (what you consume), and for project 00
   also `00-ADDENDUM.md`. Every project adopts 09's evidence contract (spec 09
   §5.14–§5.15): each decision records what it needs to be replayed.
3. The framework: the repo `README.md`, `help(decider)` and the docstrings of
   what you use (`step`, `param`, `missing_as`, `flow`, `dag`, `branch`,
   `loop`, `frame_step`, `ConfigurableStep`, `decider.steps.trees.TreeConfig`,
   `decider.steps.tables.DecisionTableConfig`,
   `decider.steps.scorecard.ScorecardConfig`, `Engine`, sessions). `IR.md` is
   the design contract. Read the framework's docs and docstrings, not its
   internals, unless you are stuck; if you had to read internals, say so in
   NOTES.md.

## Where to work

- Work in your scratch directory `<SCRATCH>/<model>/<NN-name>/` (given in your
  task). Earlier projects you consume are in sibling directories
  `<SCRATCH>/<model>/<NN-name>/`; put them on `PYTHONPATH`.
- Use the repo's environment: `uv run --project <REPO> python ...`,
  `uv run --project <REPO> pytest ...`, `uv run --project <REPO> decider ...`.
  Do not add dependencies; do not edit anything under `<REPO>/decider/`.
- **Never edit another project's directory** (including your model's earlier
  projects). If something you consume is missing or wrong, work around it in
  your own project and record it under "Gaps in what I consumed" in NOTES.md —
  whether you reused, wrapped, forked or rewrote it, and why.
- When finished, copy your whole directory to
  `<REPO>/example_projects/<NN-name>/<model>/` (create it; overwrite only your
  own directory). Do not commit.

## What to deliver (in your directory)

- A Python package for the project (a name that says what it is), with the
  decision logic as `decider` steps.
- `pipeline.py` with `build(...)` returning the pipeline step, in the shape
  `decider template` produces (run `uv run --project <REPO> decider template demo
  /tmp/somewhere` to see it), plus `inference.py` if you need a custom handler.
- `configs/<version>/` holding the params document (`params.json`) and any
  `ConfigurableStep` documents (trees, tables, scorecards) the pipeline takes.
- `sample_request.json`: one realistic single-record request.
- `SERVE.md`: the exact commands to build and serve it (including
  `PYTHONPATH` for consumed projects and the `DECIDER_*` environment
  variables: `DECIDER_API__CODE_PATH`, `DECIDER_API__PIPELINE`,
  `DECIDER_CONFIG__BASEPATH`). **Verify them**: `decider build` must succeed
  and the sample request must score through the handler (the tests in
  `<REPO>/tests/serving/` and `tests/cli/` show how to drive it without a real
  server). A project that cannot be served unchanged fails that criterion.
- `tests/`: pytest tests of the slice's behaviour, runnable with the command in
  `SERVE.md`. Run them; they must pass.
- `NOTES.md` with these sections, honestly and specifically:
  1. **What I built** — the slice, and what of the spec I left out.
  2. **Reuse** — what I reused from earlier projects (by name), what I used
     from `decider`'s built-ins (trees, tables, scorecards, params, branch,
     loop, frame steps, sessions), and what I wrote from scratch — with why.
  3. **Gaps in what I consumed** — see above.
  4. **Framework friction** — where `decider` was hard to use, missing
     something, unclear in its docs or errors, or forced a workaround. Quote
     error messages. This is the most valuable section.
  5. **Spec problems** — ambiguities, contradictions, outdated parts.
  6. **What I would do next.**

## Style

Plain, maintainable code a new engineer can follow with little context: small
functions, clear names, docstrings on the public steps saying what they decide.
No framework internals, no code generation. Keep tables, thresholds and
weights in config documents or params, not in code, where the spec says
people other than engineers change them.
