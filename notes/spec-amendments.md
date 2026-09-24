# Spec amendments agreed with the user

Changes to IR.md / Plan.md agreed after sign-off. IR.md itself is not edited;
where they disagree, this file wins.

## 2026-09-23

- **Model tiers.** Every task runs on Opus, whatever tier Plan.md names.
- **numba / polars imports (IR.md §1 point 7, §10 test 11).** Not a hard
  requirement. `steps/`, `engine/ir/` and `engine/params/` may import numba or
  polars when that is the simplest route. Acceptance test 11 is dropped.
- **Type tags live in the registry (T2.1), not T1.1.** T2.1 runs before T1.1;
  `ConfigurableStep` in T1.1 uses the registry for import-path tags, aliases,
  duplicate detection and did-you-mean.
- **`param()` without a usable default.** decider2 rejected `None` and `bool`
  defaults because `param()` returns a subclass of the default's type (so the
  bare function stays callable), and `bool`/`NoneType` can't be subclassed.
  New rule:
  - `param(default, ...)` accepts any default, including `None` and `bool`.
    For subclassable types it still returns a carrier of the default; for
    `bool`/`None` it returns a plain marker.
  - `param(required=True, ...)` declares a param with no default; missing from
    the params document means INVALID.
  - `FunctionStep.__call__` replaces any param marker left in the call with its
    default, so decorated steps are callable with defaults for every type.
- **Renames are allowed** where a decider2 name no longer fits (e.g. its
  reserved `shared` *bundle* is dropped; the top-level `"shared"` key in the
  params document stays).
- **Comments.** Code comments are minimal. Public API docstrings are user docs:
  what it does, non-obvious arguments, a short example. No references to docs,
  chapters, sections, stages or experiments. Design rationale goes here in
  `notes/`.

## 2026-09-23, after Phase 0

- **Autonomy.** The orchestrator runs every phase to completion without
  stopping at checkpoints; it still never pushes or opens PRs.
- **No code generation.** Nothing renders Python source, writes generated
  `.py` files or calls `exec`/`eval`. Fused kernels use decider2's
  `numba.extending.intrinsic` approach (fixed `kernel(n, cols, valids,
  params_all, outs)` signature, tuples for every count); trees and tables are
  data walked by generic kernels. Overrides any "real files" wording in
  Prompt.md or Design.md.
- **Performance.** A slight regression against decider2 is acceptable; a large
  one is not. T3.3 reports batch throughput and `score()` latency against
  decider2.
- **Kernel grouping default.** One kernel per innermost sequence of `scalar`
  CallNodes (a named `dag`/`flow` of plain function steps); outer sequences
  are a Python loop over kernels. Nothing wider is fused implicitly.
- **Python fallback** catches `NumbaError` and `UnsupportedBytecodeError`
  (compile-time only); runtime errors propagate.
- **Compile content key** includes constants (`co_consts`), not only
  `co_code`.
- **Trees** accept both decider_old's flat rules format and its v3 format
  through one `TreeConfig` interface. v1/v2 documents raise a clear
  deprecation error naming the version.

## 2026-09-23, T1.1b

- **`Input.arg`.** `Input` gains `arg: str`, the function argument it feeds
  (defaults to `name`). `name` is the resolved column and is the only field a
  relabel changes.
- **`CallNode.consts`.** `CallNode` gains `consts: tuple[tuple[str, Any], ...] = ()`:
  named literal arguments, e.g. a config's inline `Value[T]`. They are runtime
  arguments like params, never baked into compiled code, so changing one never
  recompiles. `params` holds only `ParamDecl`s; `to_ir` rejects anything else.
- **Call conventions** (replace the IR.md §4.2 table):
  - `scalar`: `fn(**{i.arg: value for i in inputs}, **dict(consts), **params)`.
  - `row`: always `fn(row, params, consts)`, three tuples in declared order
    (`consts` holds the values only); `reference(row, params, consts, visit)`.
    One signature for every row node, empty tuples when there is nothing.
  - `frame`: unchanged, `fn(df) -> df`.
- **`ParamDecl.schema`** is a tuple of `(column, dtype)` pairs, and `default`
  / `field_info` are left out of `==` and `hash`, so a `ParamDecl` is hashable.
  `parameters()` still reports the schema as a dict.
- **Step names** must be non-empty and contain no `/` or `#`; checked when a
  step is made or renamed and when a path is joined.
- **`IRContext.expand(owner, helper)`** is how a `ConfigurableStep` builds
  helper steps: the helper's nodes sit under the owner's name and the root
  node takes the owner's origin.
- **`IRContext` has no `shared` field** (IR.md §4.3). Shared param types are
  checked once over the finished IR in `to_ir`, which sees every node, so the
  context carries only `path` and the path → Step map it collects.
- **`engine.step_map(step)`** returns the `path → Step` map of IR.md §4.1,
  including branch conditions, arms and loop bodies. `Step.walk()` stays the
  authoring tree (named members of flows and dags).

## 2026-09-23, T2.2

- **`ParamDecl.arg`.** Like `Input.arg`: the function argument a param feeds,
  defaulting to `name`. `name` stays the key in the params document. Needed so
  a config field (`threshold`) can be fed by a local `ParamRef`
  (`{"param": "hi_cut"}`). Scalar calls pass params as `**{d.arg: value}`.
- **`ConfigurableStep.load(path_or_dict)`** loads a document by its `type` tag.

## 2026-09-23, T2.3b

- **`IRContext.call(owner, fn, inputs=None, outputs=None, values=None) -> CallNode`.**
  Builds a scalar `CallNode` from `fn`'s signature, as `step(fn)` does (types,
  null policies, multiple outputs). `inputs` maps argument -> column,
  `outputs` defaults to `(owner.name,)`, `values` maps argument -> a config's
  `Value[T]`: literals go to `consts`, `ParamRef`s become `ParamDecl`s with
  `arg` set. An argument `fn` lacks is an `IRError`. `decider.engine.ir`
  exports `CallNode`, `Input`, `Output`, `ParamDecl`, `IRContext` and the
  other node types.
- **Emit paths.** `name@path` accepts the path relative to the declaring flow
  (as before) or the absolute node path used by `step_map`, `RunReport`,
  `State` and the params document. The output column is named as written.
- **Branch/loop scopes.** Unchanged: only `modifies`/`carries` leave. New:
  reading a name after a branch or loop, when its only writer is inside that
  branch (arm or condition) or loop body, is a `WiringError` naming the
  writer and the `modifies`/`carries` list, instead of silently becoming an
  input column. A bare `emit("name")` of such a name is the same error,
  pointing at `name@<absolute path>`. Scoped intermediates nothing reads stay
  legal, and every one is emittable by path.
- **Errors.** Library errors derive from `decider.exceptions.DeciderError`
  and keep their builtin base: `WiringError(ValueError)` for wiring, names,
  paths, dag/branch/loop shape; `IRError(TypeError)` for malformed steps or IR
  (signature, condition, origin, shared-param type conflicts);
  `ParamsError(ValueError)`, `MissingInputError(ValueError)` (HTTP 400),
  `ArrowKindError(TypeError)`, `NeedsKernelSplit`, `ArrowImportError(RuntimeError)`.
  All defined in `decider.exceptions`; old import locations still work.
- **Lambda outputs.** `step(lambda ..., name="dbl")` writes `dbl`. Named
  functions keep the `(fn.__name__,)` default.
- **`ConfigurableStep.load`** also accepts JSON text (a string starting with `{`).

## 2026-09-23, T4.2 (trees)

- **`Literal[...]` string outputs.** A row node's compiled `fn` returns the
  index of a `Literal` choice; the unit decodes it to `str` (-1 = null). The
  Python `reference` returns the string itself.
- **`bytes` inputs are raw string spans** gathered through the Arrow shim
  (`(address, length)`, length -1 = null). A byte-reading row node's `str`
  params arrive as UTF-8 spans. Strings are never preprocessed into codes for
  trees.
- **Regex, case-insensitive and whitespace-trimming matches** run the tree's
  Python walker as a `Fallback` in compiled modes, on raw strings.
- **Tree literals are consts; only `ParamRef`s are params.**

## 2026-09-23, T4.3 (tables, tree nulls)

- **`DecisionTableConfig`** (alias `decision_table`): `columns`, `rows`
  (inline or `{"table": name}`), `expression`, `outputs`, `default`; first
  matching row wins; a null input never matches; decider_old's
  `"parameters": {"data", "dtypes"}` document loads as written; outputs are
  flat columns (`unnest_output` accepted and ignored).
- A String output of a table whose rows come from params is declared as an
  Enum column with its categories; such outputs are always nullable.
- **Tree null handling:** `TreeConfig.null_handling = "otherwise"` (default,
  decider_old's routing of a null feature to the otherwise branch) or
  `"error"`. String conditions keep decider_old's `match`/`no_match`/`error`.
  NaN counts as null in trees and tables.

## 2026-09-23, G1 and hot reload

- **`TreeConfig.path_output: str | None`** adds a String output naming the
  leaf that answered (per rule in `all` mode; null when the default answered).
- **Params documents are treated as immutable** once passed to `run`/`score`;
  a reused document object is hashed once.
- **Hot reload:** `Session.reload(new_pipeline)` diffs by content and re-runs
  from the first change; `Session.watch(fn)` (IPython after-cell hook) and
  `ModuleWatcher("module:attr")` (source files) call it. Serving never
  hot-reloads; it keeps explicit stage/activate.

## 2026-09-24, framework fix round

- **Frame steps take params:** `fn(df, **params)`; `param()` declarations in a
  frame function's signature work like scalar steps'.
- **One type per input column:** two steps reading one input column with
  different annotations is a `WiringError` (annotate both the same; convert
  inside the step that needs the other type).
- **Compiled modes run what they can't compile in Python:** a step compiled
  modes refuse (e.g. comparing two `str` inputs) runs as a per-step Python
  fallback with a one-time warning; `SteppedRunner(strict=True)` raises instead.
- **Serving:** `code_path` always first on `sys.path`; warm-up uses
  `sample_request.json` when present; JSON inputs are coerced to declared
  `date`/`datetime`/`list`/TypedDict annotations; `RequestHandler.warm_fn` is
  overridable; `from decider.serving import RequestHandler`.
## 2026-09-24, trees, tables and registry fixes

- **`TreeConfig.trace_output: str | None`** adds a String output holding the
  ordered path, the ids of every node walked joined with `>`, from both
  walkers in every mode (per rule in `all` mode; in `first_match`, the path
  in the rule that answered, else the last rule's).
- **Decision-table band ladders are per `eq` group.** Rows with equal `eq`
  column values form one ladder: open edges (`None`), neighbour fill and
  contiguity apply within it.
- **Built-in tags resolve lazily.** `tree`, `decision_table` and `scorecard`
  resolve without their modules imported first (`BaseRegistryModule.lazy`).

## 2026-09-24, fix-round follow-ups (FIX-H)

- **`Engine(strict_compile=True)`** passes `strict` to the stepped and fused
  runners.
- **`round(x, n)` in kernels equals CPython's** for float `x` and
  `|n| <= 22` (exact half-to-even on the double's value); beyond 22 digits it
  scales and rounds. **`x ** n`** (float `x`, int `n`) calls libm `pow` like
  CPython instead of multiplying by squaring (a literal `x ** 2` still
  compiles to `x * x`).
- **Typed frame reads:** `frame_step(reads={"accounts": list[Account]})`
  gives read columns types, like a plain step's annotations (JSON date
  coercion, one type per input column). A list keeps them untyped.
- **An input column is described by its first typed reader**, so an untyped
  frame step reading it first no longer hides a later step's `date`.
- **JSON coercion keeps undeclared TypedDict keys**: a TypedDict naming only
  the date fields leaves the rest of each dict as sent.

## 2026-09-24, parameter tables for plain steps (FIX-G)

- **`param_table(columns, default=[...] | required=True, shared_key=None)`**
  (`from decider import Table, param_table`) declares a table-valued param in
  a plain step's signature, like `param()`: `rates: Table = param_table({"floor":
  int, "rate": float}, default=[...])`. Columns are `int`, `float` or `bool`
  and must be identifiers (they become namedtuple fields); string columns are
  refused (use a `DecisionTableConfig`, or code the key as an int).
- **Representation:** the step receives a namedtuple (`bundle_class` of the
  column names, so numba's disk cache can pickle it) of read-only 1-D numpy
  arrays (int64, float64, bool), in every mode. It is a runtime argument in
  the params bundle, never a constant, so row edits and row counts keep one
  numba type; only a column change recompiles. `Table` is `Any`, for readers.
  A direct call of the plain function gets the default table itself.
- **Validation:** the `ParamDecl`'s annotation is `table_type(schema)`, a
  cached `Annotated[list[TypedDict], AfterValidator(to columns)]` (strict,
  extra keys forbidden), one object per schema so shared tables compare
  equal. Errors read `<path>: param 'rates': row 1, column 'floor': ...;
  expected a list of rows like [{"floor": int, "rate": float}, ...]`.
- **`parameters()`** reports every table as `{"type": "table", "schema": ...}`
  plus `"default": rows` or `"required": True` (tables of a
  `DecisionTableConfig` too); **`defaults()`** shows a required table as `[]`
  instead of leaving it out; **`json_schema()`** gives each table's column
  types, `additionalProperties: false` and its default rows.
- **Params errors:** a nested pydantic `missing` (a row lacking a column) is no
  longer reported as "param is required but missing"; a missing required
  table's error names its expected rows. `ParamRef.param` and `TableRef.table`
  must be identifiers (not keywords, no leading `_`); a bad one fails at load
  with a suggestion (`hi-cut` -> `hi_cut`).
- Interpreted mode indexes numpy arrays, so a table value is a numpy scalar
  there (`x / 0.0` on one gives `inf` with a warning where a kernel raises).
