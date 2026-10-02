# Flow files: structure in YAML/JSON, steps in Python

**Decision:** a flow file (`flow.yaml`, `flows/<name>.yaml`, or `.json` with the
same schema) is a front end. `decider.flowfile.load_flow` turns it into the same
`flow`/`branch`/`loop` objects written by hand, so the engine, compiler, serving,
debugger and experiments see an ordinary `Step` and change nothing.

Modelled on the granting-decisioning service's `flow.yaml` (a tree of
`step`/`subflow`/`when`/`loop` nodes naming code steps), without its problems:
subflows there are copy-pasted between flow variants (BureauArods appears in
three files) and `when`/`loop` take free-text expressions.

## What it admits

- `step: name` resolves to a function or `Step` in the `components` package, or
  to a config document of that name in the active config version (a scorecard,
  decision table or tree, loaded by its `type`). Both is an error, never a
  precedence rule.
- `flow: name` with `steps:` is an inline named flow; without, it loads
  `flows/name.*`, so subflows are shared by reference instead of copied.
- `branch` with no `else` is a one-arm `branch()`: false rows pass through. We
  allowed one arm in `branch()` rather than adding a `when()` primitive; the
  engine already ran an empty arm in every mode.
- Conditions are a bool column, a component step, or `{expr: "..."}`.
  `expr_condition` translates the expression into a one-rule tree document, so it
  reuses the tree kernel (numeric and exact-string matches compile; no new
  evaluator). Nulls make it false, under negation too.
- `modifies` and `carries` stay required, as in Python.

## Why this doesn't reopen `structure-in-python.md`

That decision was against two copies of the structure drifting apart. A flow file
is the only copy, and it lives with the code: reviewed and deployed with the
components it names. Params and step documents stay versioned in `configs/`. The
`components:` import path is therefore a pointer in reviewed source, not in
deployable config.

Not built: a `FlowConfig(ConfigurableStep)` that could live inside a config
version, i.e. structure that changes without a deploy. Add it only if that is
wanted, and treat it as reversing the earlier decision.

## Deferred

- `cases:` (multi-way branch on values): nest branches with `expr` conditions.
- `dag:`, `each:`, `emit`/`drop` on flow nodes.
- An id writer for flow files; ids are written by hand (`id:`) for now.
- `a OP b` with no constant side compiles as `(a) - (b) OP 0` in float, so exact
  equality of large ints can be off.
