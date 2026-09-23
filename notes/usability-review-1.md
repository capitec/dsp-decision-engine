# Usability review 1 (T2.3, fresh eyes)

I learned the API only from `help()`, `inspect.signature`, `__doc__` and `dir()`. I did not open any source files, design docs, tests or notes. Scripts: scratchpad `usability/p1*.py`, `p2*.py`, `p3*.py`, `rules.py`.
Severity: **blocker** (a new user can't proceed without help), **friction** (costs time or misleads), **nit**.

## Pipeline 1: function steps, `param`, `missing_as`, `dag`/`flow`, params document

Built `dag(affordable, ratio, disposable_income)` piped into `score(bureau_score = missing_as(0.0), weight = param(0.01))`, then ran it on polars.
**What worked:** functions stay plain Python. `parameters().defaults()` gives a complete nested document that you can edit and pass back. Partial documents are accepted. Int columns cast into float inputs. `dag` cycles and two-writer conflicts get clear errors. Param-path typos get a did-you-mean (`no step with params at 'app/scor'. Did you mean 'app/score'?`). `MissingInputError` explains how to fix itself (`declare missing_as(fill) or T | None`).

- **friction:** intermediate outputs (`ratio`, `disposable_income`) silently disappear from the output. `Executable.run` says "minus drops" but never says that intermediates are dropped by default. See top-5 #2.
- **friction:** `step(lambda x: x*2, name="dbl")` writes a column named `<lambda>`: `name=` sets the path but not the output. *Suggest:* with a lambda, default `output` to `name`, or raise an error that asks for `output=`.
- **nit:** the `ParamsError` header reads `invalid params (2 rows):`. The count is data rows, which you can only work out by experiment, and `(1 rows)` is ungrammatical. *Suggest:* `invalid params (affects 2 of 2 rows):`.
- **nit:** `help(decider)` is about 900 lines, and about 750 of them are pydantic `BaseModel` methods inherited by `ConfigurableStep`. The package has no module docstring or quickstart.

## Pipeline 2: `branch`, waterfall, emit, `score`, engine modes

Built `flow(term_cap, cap_by_income, branch(is_private, cap_private, cap_public, modifies=["term_cap"]))`. The brief expected a top-level `emit("x@*")`. That doesn't exist. `emit` is a method on `SequentialStep`/`DagStep` only (not on `Step` or `FunctionStep`), and I found it only through `help(type(flow(...)))`.
**What worked:** `term.emit("term_cap@*")` shows every version in the waterfall, with nulls for the arm a row didn't take, plus the merged `term_cap@by_sector`. Emit typos get did-you-mean. `Engine().bind(p).score({...})` returns a dict and accepts `params=`. `RunReport` works. Lazy validation skips unreached nodes, and `on_invalid="warn"` records a warning. `.bind(cap=…)`, `.relabel`, `.drop` and the `.named()` hint on duplicate paths all behave as documented.

- **friction:** emit paths are **relative to the flow** (`term_cap@cap_by_income`), while the params document, `RunReport`, `step_map`, `State.versions` and the `State` docstring example all use **absolute** paths (`term/cap_by_income`). Copying a path from `exe.report` into `emit` fails. The error does list the valid producers, which helps.
- **friction:** a branch arm that writes a name not in `modifies` (`flag`) is silently discarded, and `emit("flag")` then says "no step produces 'flag'". The branch condition's output (`is_private`) can't be emitted either. The reverse mistake is already an error (`modifies 'x', but no arm writes it`). *Suggest:* make this an error too, and make condition outputs emittable.
- **nit:** a bool condition sends True to arm 0, but an int condition picks arm `i`, so `int(True) == 1` points at the other arm. That is easy to trip on when you switch a flag from bool to 0/1. *Suggest:* say so in the `branch` docstring.
- **friction:** `Engine.bind(mode=...)` has only `"interpreted"`, but it is typed `str`. It should be a `Literal`. `Step.session()` is listed on every step but raises `NotImplementedError: debug sessions are not built yet`. *Suggest:* hide it until it exists.
- **friction:** exceptions from user code (`ZeroDivisionError`) come out raw, with no step path, row index or inputs.
- **nit:** `score()` on a dict with a missing key says `column 'net_income' is not in the input frame`. The user passed a record, not a frame.

## Pipeline 3: custom `ConfigurableStep`, JSON round trip, tree schema

Wrote `ThresholdRule(column, threshold: Value[float])` and loaded it with `{"threshold": {"param": "min_ratio", "default": 2.0}}`. Composed it as `flow(ratio, rule, decision)`, dumped it, loaded it back and ran it.
**What worked:** the param shows up in `defaults()` under the rule's name as `min_ratio`, and retuning it works without a rebuild. `model_dump_json()` → file → `ConfigurableStep.load(path)` gives an equal object that runs. Loading by import path (`"rules:ThresholdRule"`) works. An unknown `type` gets a did-you-mean. Refs with `shared` and required refs behave as documented.

- **blocker (for a first user):** the `to_ir` example uses `CallNode`, `Input`, `Output` and `ParamDecl` with no imports. None of them is exported from `decider`, `decider.engine` or `decider.engine.ir`. I found them by running `dir()` over `decider.engine.ir.nodes`. Even then, every author has to copy the `isinstance(factor, ParamDecl)` params-or-consts split by hand. See top-5 #1.
- **friction:** `ConfigurableStep.load(rule.model_dump_json())` gives `FileNotFoundError` with the whole JSON document as the filename. *Suggest:* treat a string starting with `{` as JSON text, or say "expected a path or a dict".
- **friction:** the `load` docstring example uses `TreeConfig.load("trees/risk.json")`, but `TreeConfig` doesn't exist. `decider.steps.trees` calls itself "Decision trees as configurable steps", but it contains no step.
- **nit:** retuning with the argument name (`{"threshold": 10}` instead of `min_ratio`) gives `unknown param 'threshold'` with no hint. *Suggest:* list the valid params, as `.bind()` already does (`its params are ['cap']`).
- **nit:** dumps include `"shared": false`, and a config's repr shows `reads=() writes=()`. That's noise in a document meant for editing.
- **Trees:** `load_document(doc).to_tree()` accepts `{"param": "young_age", "default": 30}` as a threshold in both `flat_rule` and `v3` documents. `required_params()` returns `{'young_age'}`, names only. *Suggest:* return the refs so defaults are visible. No tree `ConfigurableStep` is registered yet, so a tree can't be run. Validation errors have the internal title `function-before[_supported(), tagged-union[function-after[_normalise(), V3TreeDocument],…]]`. The field path under it is good. *Suggest:* use a `TreeDocumentError("flat_rule document: …")` wrapper instead.

## Cross-cutting

- **friction:** there is no common error base. `ParamsError` (`decider.engine.params`, a `ValueError`), `MissingInputError` (`decider.engine.boundary.nulls`) and `ArrowKindError` (private `…_arrow.plan`, a `TypeError`) each live somewhere different. `decider.exceptions.DeciderError` exists, but none of them inherit from it. You can't write `except decider.DeciderError`.
- **friction:** `Engine`, `Value` and `ParamRef` are not top-level exports, so you have to guess `decider.engine` and `decider.steps`.

## Top 5, ranked

1. **Make `ConfigurableStep.to_ir` authorable from the public API** (blocker). Export `CallNode`, `Input`, `Output` and `ParamDecl` from `decider.engine`, and put the imports in the docstring. Better still, add `ctx.call(fn, inputs={"x": self.column}, outputs={self.name: bool}, values={"threshold": self.threshold})` so the literal-vs-param split is handled once, inside the library.
2. **Make emit discoverable and paths consistent** (friction). Document the drop-intermediates default and `.emit`/`.drop` in `help(flow)` and `help(dag)`. Put `emit` on `Step` (or export a top-level `emit`). Accept the absolute paths that `report`, `step_map`, `State` and the params document use.
3. **Stop silently discarding branch-arm writes.** Raise an error when an arm writes a name not in `modifies`, and make condition outputs emittable.
4. **Give errors one base and add context.** Have every library error inherit from `decider.exceptions.DeciderError` and re-export it there. Wrap exceptions raised by user functions with the step path, row index and input values, chaining the original.
5. **Clean up the front door.** Add a package docstring with a 15-line quickstart. Export `Engine`, `Value` and `ParamRef` from the top level. Hide `Step.session()` until it works. Type `bind(mode=)` as a `Literal`. Default a lambda step's output to its `name`. Let `ConfigurableStep.load` accept JSON text. Fix the `TreeConfig` example.

## Triage (T2.3b)

| Finding | Outcome |
|---|---|
| IR classes not exported; params/consts split by hand (blocker) | **Fixed.** `decider.engine.ir` exports `CallNode`, `Input`, `Output`, `ParamDecl`, `IRContext` and the node types. `ctx.call(self, fn, inputs=..., outputs=..., values=...)` reads `fn`'s signature and splits literals/refs once. `ConfigurableStep` docstring and the `ThresholdRule` test use it with public imports only. |
| Intermediates silently dropped; `emit`/`drop` undiscoverable | **Fixed.** `flow`, `dag` and the package docstring explain the default and show `.emit`/`.drop`. |
| `emit` only on flows/dags, no top-level `emit` | **Deferred.** `flow(step).emit(...)` covers it; add `Step.emit` if users still miss it. |
| Emit paths relative, everything else absolute | **Fixed.** `name@path` takes either; did-you-mean covers both forms. |
| Arm write outside `modifies` silently lost; condition output not emittable | **Fixed.** Reading such a name after the branch (or loop), or bare-emitting it, is a `WiringError` naming the writer and `modifies`/`carries`; `name@path` emits arm writes and the condition output (it already could, now documented). |
| Arm write shadowing a name that exists before the branch | **Deferred.** Still reads the pre-branch value; the brief limited the error to the only-producer case. Revisit if it bites. |
| bool vs int condition arm order | **Fixed** (documented in `branch`). |
| No common error base | **Fixed.** `WiringError`, `IRError`, `ParamsError`, `MissingInputError`, `ArrowKindError`/`NeedsKernelSplit`, `ArrowImportError` derive from `DeciderError` and keep their builtin base; all in `decider.exceptions`. |
| Errors left on builtins | **Deferred.** API-argument misuse (`step(output=, outputs=)`, `param()`, `missing_as(None)`, `as_step`) stays `TypeError` like Python's own; registry load errors (`decider/registry`), `Engine.bind` unknown mode and input-shadowing `ValueError`s (`engine.py`, being edited by the compiled-modes task) and `ExprError` (`steps/expr`) were outside this task's files. |
| User-function exceptions lack step path/row/inputs | **Deferred** to after T3.3 (touches runners). |
| `Engine`, `Value`, `ParamRef` not top-level | **Fixed.** |
| No package docstring; `help(decider)` ~900 lines | **Fixed** (quickstart docstring now leads `help(decider)`). The pydantic methods of `ConfigurableStep`/`ParamRef` still follow it; `help(decider.flow)` etc. are short. |
| `bind(mode=)` typed `str` | **Deferred.** `engine.py` belongs to the compiled-modes task, which adds the modes; one-line `Literal` there. |
| `Step.session()` raises `NotImplementedError` | **Resolved** by T1.5 (sessions exist). |
| `step(lambda, name="dbl")` writes `<lambda>` | **Fixed** for lambdas; named functions keep `fn.__name__` as IR.md pins. |
| `ConfigurableStep.load(json_text)` → `FileNotFoundError` | **Fixed.** |
| `TreeConfig.load` example; trees module claims a step | **Fixed** (example replaced; trees docstring describes documents only). |
| `ParamsError` header `(2 rows)` | **Fixed:** `(affects 2 rows)` / `(affects 1 row)`. |
| Retuning by argument name gives no hint | **Fixed:** points at the document key (`did you mean 'min_ratio' (it feeds argument 'threshold')?`), otherwise lists the node's params. |
| `score()` missing key says "input frame" | **Fixed:** "not in the input frame or record". |
| Config repr shows `reads=() writes=()` | **Fixed** (`repr=False`). |
| Dumps include `"shared": false` | **Deferred.** Needs `Field(exclude_if=...)` (pydantic ≥ 2.12, we pin `>=2`) or a custom serializer; noise only. |
| Trees: `required_params()` names only; union-title validation errors; no tree step | **Deferred** to T4.2 (tree step). |
