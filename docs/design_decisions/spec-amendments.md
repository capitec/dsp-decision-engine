# Amendments to IR.md

Changes to `IR.md` that override it: read this next to `IR.md` before relying on a section it names, because where the two disagree this file is right.

## IR.md §1 The two layers

- "Nothing in either layer imports numba or polars" (point 7) does not hold. `steps/`, `_engine/ir/` and `_engine/params/` may import them when that is simplest (the decision table step uses polars). The matching acceptance test is gone.
- The IR is not closed at four node types: `ScatterGatherNode` (`each`) and `FiniteStateMachineNode` sit beside `CallNode`, `SequenceNode`, `BranchNode` and `LoopNode`. `decider.ir.iter_nodes` walks them all.
- `decider.engine.ir` is now `decider._engine` (private); the public IR names, `to_ir` and `step_map` are exported from `decider.ir`.

## IR.md §3 Steps

- `param(default, ...)` accepts any default, `None` and `bool` included, and `param(required=True, ...)` declares a param with no default (missing from the params document is invalid). `FunctionStep.__call__` swaps a param marker left in the call for its default.
- `FunctionStep` outputs: a lambda takes `name=` as its output (`step(lambda ..., name="dbl")` writes `dbl`); a named function keeps `(fn.__name__,)`.
- `Step.walk()` is still the authoring tree (named members of flows and dags). The path to step map is `step_map(step)`, which also covers branch conditions, arms and loop bodies.
- A step name is non-empty and has no `/` or `#`; this is checked when a step is made or renamed and when a path is joined.
- A frame step takes params: `fn(df, **params)`, with `param()` declarations in its signature like a scalar step's. `frame_step(reads={"accounts": list[Account]})` types the read columns like a plain step's annotations; a list leaves them untyped.
- Wiring (all `WiringError`): two steps reading one input column with different annotations (an input column is described by its first typed reader, so an untyped frame step first does not hide a later `date`); reading a name after a branch or loop when its only writer is inside that arm, condition or body (only `modifies` and `carries` leave; the message names `name@<absolute path>`). A scoped intermediate nothing reads stays legal.
- `emit`/`drop` accept `name@path` with the path relative to the declaring flow or the absolute node path. The output column is named as written.
- Every step constructor and decorator takes `id=` (12 lowercase hex characters, committed in source); `ConfigurableStep` has an `"id"` field next to `name`. An id never changes name, path, `step_map` or execution. `dag(single)` collapses only when both `name` and `id` are `None`. `decider ids` writes them into source (clean tracked tree only).
- `ConfigurableStep.load(source)` takes a path, a dict or JSON text (a string starting with `{`) and picks the class by its `type` tag. Built-in tags (`tree`, `decision_table`, `scorecard`) resolve without importing their modules first.
- Trees accept decider 0.3 flat rules and the v3 format through one `TreeConfig`; v1/v2 documents raise a deprecation error naming the version. Tree literals are consts, only `ParamRef`s are params, and a `ParamRef.param` / `TableRef.table` must be an identifier (not a keyword, no leading `_`), else it fails at load with a suggestion.
- `TreeConfig.null_handling`: `"otherwise"` (default, a null feature takes the otherwise branch) or `"error"` (`MissingInputError`); string matches have their own `no_match` / `match` / `error`. NaN counts as null in trees and tables. `path_output` adds a String column naming the answering leaf (null when the default row answered); `trace_output` adds the ids of every node walked joined with `>`. Both are per rule in `mode: "all"`.
- A tree's regex, case-insensitive and whitespace-trimming string matches run the Python walker as a `Fallback` in compiled modes, on raw strings.
- `DecisionTableConfig` (`decision_table`): first matching row wins; a null input never matches; a decider 0.3 `"parameters": {"data", "dtypes"}` document loads as written (`unnest_output` is ignored); outputs are flat columns. Band ladders are per `eq` group: rows with equal `eq` values form one ladder (open edges, neighbour fill, contiguity). A String output of a table whose rows come from params is a nullable Enum column.

## IR.md §4.1 `IRNode` and `Origin`

- `Origin` has two more fields: `id: str | None` (the durable step id) and `suppress_checks: tuple[str, ...]` (check codes the step suppresses; metadata, never part of a cache key or the IR fingerprint). `check_id()` validates `[0-9a-f]{12}`.
- Identity: `path` and `name` are derived ids for display; `id` is the committed durable one. There is one id per definition and many per path (a reused step keeps its id at each placement). `(flow_id, step_id)` is the global reference; anonymous flows have no id.

## IR.md §4.2 The node types

- `Input` and `ParamDecl` have `arg`: the function argument they feed, default `name`. `name` is the resolved column (or the params-document key) and is all a relabel changes. This lets `{"param": "hi_cut"}` feed a `threshold` argument.
- `CallNode.consts: tuple[tuple[str, Any], ...] = ()` holds named literal arguments (a config's inline `Value[T]`). They are runtime arguments like params, never baked into compiled code, so changing one never recompiles. `params` holds only `ParamDecl`s; `to_ir` rejects anything else.
- `ParamDecl.schema` is a tuple of `(column, dtype)` pairs and `default` / `field_info` are out of `==` and `hash`, so a `ParamDecl` is hashable. `parameters()` still reports the schema as a dict.
- Call conventions replace the table in IR.md:
  - `scalar`: `fn(**{i.arg: value for i in inputs}, **dict(consts), **params)`.
  - `row`: always `fn(row, params, consts)`, three tuples in declared order (`consts` holds values only); `reference(row, params, consts, visit)` is the Python twin.
  - `frame`: `fn(df) -> df`, unchanged.
- A row node's `Literal[...]` string output is the index of the choice, decoded to `str` by the unit (-1 is null); the Python `reference` returns the string. A function step's own `Literal` output runs in Python.
- Strings: a `bytes` or `Text` input is an `(address, length)` span gathered through the Arrow shim (length -1 is null); `Raw[str]` is now `Text`, `Rows[Item]` is now `Columnar[Item]`. A semantic `str` step runs in Python row by row (`dispatcher.py_func`), because one compiled call per row costs more than the Python body. Strings are never preprocessed into codes.
- Compiled code: no source is generated and nothing calls `exec`/`eval`. A fused kernel is built by a numba `intrinsic` with the fixed signature `kernel(n, cols, valids, params, flats, sinks, outs[, header, cursor, offsets])` (the last three only when tracing), using tuples for every count. Kernels are `nogil=True` and `fastmath` is off; `round(x, n)` and `x ** n` equal CPython's.
- Kernel grouping: one kernel per innermost run of `scalar` CallNodes; outer sequences are a Python loop over kernels. Nothing wider fuses implicitly.
- The compile key is `fingerprint(fn)`: bytecode with `co_consts`, argument layout, closure cell values and the globals it reads, never name, file or line. The disk index key adds `SALT` and that fingerprint.
- Python fallback, per step: compiling catches numba errors (`NumbaError`, `UnsupportedBytecodeError`) plus `NotImplementedError` and `AssertionError` around `dispatcher.compile` (`_COMPILE_ERRORS` in `_engine/compile/njit.py`); runtime errors always propagate. A fallback warns once with `FallbackWarning` (a `UserWarning`); `Engine(strict_compile=True)` raises instead, in the stepped and fused runners.
- `@allow_fallback` (replacing `python_only`) accepts a step's fallback: no warning, and `strict_compile` allows it; `Executable.fallbacks()` still lists it, reason prefixed `@allow_fallback: `. On a function a step calls it marks an intentional Python boundary. It takes a plain function or `step()` of one, else `TypeError`. `compile_call` returns `(key, fn, reason, declared)`.

## IR.md §4.3 `IRContext`

- `IRContext` has no `shared` field, only `path` and the path to `Step` map it collects. Shared param types are checked once over the finished IR in `to_ir`.
- `ctx.expand(owner, helper)` is how a `ConfigurableStep` builds helper steps: the helper's nodes sit under the owner's name and the root node takes the owner's origin.
- `ctx.call(owner, fn, inputs=None, outputs=None, values=None) -> CallNode` builds a `scalar` node from `fn`'s signature as `step(fn)` does. `inputs` maps argument to column, `outputs` defaults to `(owner.name,)`, `values` maps argument to a `Value[T]`: literals go to `consts`, `ParamRef`s become `ParamDecl`s with `arg` set. An argument `fn` lacks is an `IRError`.

## IR.md §5 Inputs and params

- Params documents are immutable once passed to `run` / `score`; a reused document object is hashed once.
- `param_table(columns, default=[...] | required=True, shared_key=None)` declares a table param in a plain step's signature, annotated `ParamTable` (was `Table`). Columns are `int`, `float` or `bool` and valid identifiers; string columns are refused (use a `DecisionTableConfig` or an int key).
- The step receives a namedtuple of read-only 1-D numpy arrays, in every mode. It is a runtime argument in the params bundle, so editing rows or their count never recompiles; only changing columns does. A direct call of the plain function gets the default table. Interpreted mode indexes numpy arrays, so a table value is a numpy scalar there (`x / 0.0` gives `inf` with a warning where a kernel raises).
- `parameters()` reports a table as `{"type": "table", "schema": ..., "default": rows | "required": True}`, decision-table tables included; `defaults()` shows a required table as `[]`; `json_schema()` gives column types, `additionalProperties: false` and default rows.
- A nested pydantic `missing` (a row lacking a column) is not reported as "param is required but missing"; a missing required table's error names its expected rows.

## IR.md §6 Names, paths and sources

- `Origin.path` is also the key in `step_map`, `RunReport`, `State` and the params document.

## Beyond IR.md

- Errors derive from `decider.exceptions.DeciderError` and keep a builtin base: `WiringError(ValueError)` (wiring, names, paths, dag/branch/loop shape), `IRError(TypeError)` (malformed steps or IR: signature, condition, origin, shared-param type conflicts), `ParamsError(ValueError)`, `MissingInputError(ValueError)` (HTTP 400), `ArrowKindError(TypeError)`, `NeedsKernelSplit`, `ArrowImportError(RuntimeError)`.
- Hot reload: `Session.reload(new_pipeline)` diffs by content and re-runs from the first change; `Session.watch(source)` (IPython after-cell hook) and `ModuleWatcher("module:attr")` (source files) call it. Serving never hot-reloads; it keeps explicit stage and activate.
- Serving: `code_path` goes first on `sys.path`; warm-up uses `sample_request.json` when present; `RequestHandler.warm_fn` can be overridden; JSON inputs are coerced to declared `date`, `datetime`, `list` and `TypedDict` annotations, keeping undeclared TypedDict keys.
- `decider.lab.checks` is the one checks package: `checks.run(pipeline)` uses the project's `checks.yaml`, else `checks.standard`. Checks are registry modules with stable codes (`checks.catalogue()`); findings carry `StepRef`s.
- Bridge: `python -m decider._tooling.bridge`. A `{"cmd": "hello"}` handshake replies `Hello` (`protocol`, `contract_version`, `capabilities`); replies are typed models in `protocol.py`. `FieldMeta` has `precision`, `scale` and `units` instead of `cents`; `Decimal[p, s]` and `Currency` report `kind: "decimal"`. `describe` payloads and contract step references carry `flow_id` (and `step_id`), the durable pair.
