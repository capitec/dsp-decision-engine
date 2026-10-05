# Checks

## What is wrong today

- **Two packages for one feature.**
  - `decider.check` holds `run`, `Finding`, `Report`, `Severity`, and the `Check` / `Suite` type aliases.
  - `decider.checks` holds `wall_clock`, `impure`, `numeric`, `durable_ids`, `default_suite`, `equivalence`, `compare` and `Equivalence`.
  - They import each other: `check/run.py:44` lazily imports `checks`, and every module in `checks/` imports `check`.
- **`checks/equivalence.py` is not a check.**
  - `equivalence()` runs every execution mode over a corpus, which is a test.
  - `compare()` diffs two versions, which belongs in `diff.md`, and its "structural proof" ignores constants.
  - Both pull `decider.testing` and the debugger into checks.
- **A check is a plain callable.** That means:
  - No stable code, configuration, suppression or listing.
  - It can't be named from JSON, VS Code or MCP.
  - The one configurable built-in is a closure factory (`durable_ids(severity)`) that sets `__name__` by hand.
- **`suites=` takes a sequence of sequences.** To add one check you write `suites=[checks.default_suite, (mine,)]`.
- **There are three report shapes.** `check.Report`, `data.PreflightReport` (free-text `errors: tuple[str]`) and `data.ScopeCheck`. MCP returns only the default suite (`mcp/tools.py:120-122`).
- **IR is rebuilt per check.** Every check calls `to_ir(step)` again, and so does `run` itself.
- **Findings carry a weaker `StepRef`.** `checks/_common.step_ref` re-implements `contract.describe._node_ref` without kind, inputs or outputs, so a client can't join a finding to a `describe()` node.
- **The checks are thin.** `numeric` is pure AST with no dtype knowledge: it flags `cents * rate / 12` (a false positive) and misses `a / b` and bare `round()` (see R2 in `user-stories.md`).
- **There is no CLI and no gate.** CI has to write Python.

## Target: one `decider.checks` package

```
decider/checks/
  __init__.py     Check CheckContext Finding Severity Report run standard catalogue check(decorator)
  _model.py       Severity Finding Report SkippedCheck Suppression
  _base.py        Check (registry root), CheckContext, the @check decorator
  _run.py         run(): select, configure, suppress, collect
  _report.py      to_text / to_json / to_sarif
  _suppress.py    inline comments + config suppressions
  _ast.py         shared step-source walker (source, lines, names bound to IR annotations)
  builtin/        one family per file, each under 500 lines
    determinism.py numeric.py structure.py params.py ids.py data.py runtime.py
```

`decider.check` is deleted. That means updating the GUIDE, `mcp/tools.py`, `tests/check/` → `tests/checks/`, and VS Code decision 13 through `spec-amendments.md`.

### A check is a registry module

Checks join the six existing registry roots. They then get naming from JSON, lazy built-ins, import-path lookup that only returns registered subclasses, and `json_schema()` for VS Code forms, all without new machinery.

```python
class Check(BaseRegistryModule, root=True):
    """One diagnostic. Subclass with a `type` alias; config names it by alias or "module:Class"."""
    code: ClassVar[str]                     # "DCD101"; stable, documented, never reused
    needs_data: ClassVar[bool] = False      # skipped (and reported as skipped) without a frame
    executes: ClassVar[bool] = False        # runs the flow; opt-in, not in `standard`
    severity: Severity = Severity.WARNING   # default; overridable per instance / config
    def run(self, ctx: CheckContext) -> Iterable[Finding]: ...
```

`CheckContext` is built once per `checks.run` and shared by every check. Its attributes are lazy and cached:

| Attribute | What it is |
|---|---|
| `ctx.ir` | `to_ir(step)`, built once |
| `ctx.plan` | the wiring plan: reads, producers, versions, inputs |
| `ctx.description` | `describe(step)`; findings use its `StepRef`s, so they join with VS Code nodes |
| `ctx.params` | the params document, if given |
| `ctx.frame` | the data, if given |
| `ctx.calls()` | the `CallNode`s of **authored** code. It skips generated kernels from `decider.steps.*`, where AST checks are noise. |
| `ctx.source(node)` | parsed source, line map, and argument names bound to their IR annotations, so `cents` is known to be `Money(cents=True)` int |
| `ctx.finding(node, message, fix=..., **detail)` | builds a `Finding` with code, check, step ref, file and line filled in |

There is a cheap path for a one-off rule, so nobody has to write a class:

```python
@checks.check(code="ACME002", severity="warning")
def no_print(ctx):
    for node in ctx.calls(): ...
```

### Config-driven step types check themselves

AST checks can't see inside a tree, table or scorecard. Instead, `ConfigurableStep` gets an optional `check(ctx) -> Iterable[Finding]` hook that works on the pydantic document. Examples: tree leaves shadowed by an earlier range (the validators in `steps/trees/schema/conditions.py` already exist), overlapping table rows, and scorecard bins with gaps. `checks.run` calls the hook for every configurable node. A user's own `ConfigurableStep` gets the same seam for free.

## The `Finding` standard

```python
class Finding(BaseModel):
    code: str                   # "DCD101": what CI pins and users suppress
    check: str                  # alias, "int64_overflow": human and config key
    severity: Severity          # after config overrides
    message: str                # one sentence, says what is wrong
    step: StepRef | None        # the contract ref: flow_id, step_id, path
    file: str | None; line: int | None
    fix: str | None             # one line: what to do instead
    detail: dict                # machine-readable (expression, operands, bounds)
    suppressed: Suppression | None = None   # kept in the report, excluded from `ok`

class Report(BaseModel):
    findings: tuple[Finding, ...]; skipped: tuple[SkippedCheck, ...]
    ok: bool; counts: dict[Severity, int]
    def fail_on(self, severity) -> bool
    def to_text(self) -> str; def to_sarif(self) -> dict  # JSON is model_dump_json
```

Rules for writing checks:
- **Codes** are `DCD` followed by a family digit: 0 determinism, 1 numeric, 2 structure, 3 params, 4 ids and governance, 5 data and runtime. Codes are never reused. A third-party check uses its own prefix.
- **Messages say what and where. `fix` says what to do.** Neither restates the other.
- **Decider reports and the client decides.** `Report.ok` and `--fail-on` are the gate. Default severity is advice, not policy.
- **No silent skips.** A check that needs data or execution and wasn't given it appears in `report.skipped` with the reason.
- **Locations are contract refs**, so a finding survives renames and links straight to the editor.
- **Findings come from one `StepRef` builder.** That is `contract.describe`; `_common.step_ref` goes.

## Configuration and suppression

```json
{
  "extends": "standard",
  "checks": [
    {"type": "int64_overflow", "severity": "error"},
    {"type": "durable_ids", "severity": "error"},
    {"type": "acme.checks:NoFloatMoney", "allow": ["fee"]}
  ],
  "disable": ["DCD107"],
  "suppress": [{"code": "DCD101", "step_id": "0123abcdef45", "reason": "capped upstream"}]
}
```

Suppression works at three levels, from narrowest to widest:
1. **Inline:** `# decider: ignore[DCD101] reason`. A bare `ignore` with no code is rejected.
2. **By durable step id in config.** This survives refactors and gets reviewed in the PR.
3. **`disable` or a severity override for a whole check.**

Where the file lives is decision 3, and whether inline comments are allowed is decision 4 (both in the README).

## Running

```python
from decider import checks
report = checks.run(pipeline)                                        # checks.standard
report = checks.run(pipeline, [checks.standard, NoFloatMoney()])     # flat list of checks or check sets
report = checks.run(pipeline, config="checks.json", data=frame, params=doc)
checks.catalogue()                                                   # alias, code, family, needs_data, severity, schema
```

```bash
decider check credit.pipeline:build [--config checks.json] [--data loans|file.parquet]
              [--format text|json|sarif] [--fail-on error|warning] [--select DCD1] [--list]
```

What VS Code and MCP get:
- `check_report(entry, config=None, data=None)` and `checks_list()` as MCP tools.
- `Check.json_schema()` for a settings form.
- SARIF output, which the stock VS Code SARIF viewer and GitHub code scanning both read. Its `partialFingerprints` is `flow_id/step_id/code`, so results track across edits.

## Standard catalogue

`standard` holds the static checks: no data, no execution. Data and executing checks are opt-in. For the build order, the money-safety ones come first; `money-rounding-and-overflow.md` already asks for them.

What the IR gives a check is more than `numeric` uses today:
- `Input`, `Output` and `ParamDecl` annotations, with `metadata_of` for `Money` / `Percent` / `Duration`.
- `ParamDecl.field_info` bounds, and `FeatureKind` per value.
- `Plan` reads, producers and versions.
- Branch conditions, loop carries and `max_iterations`.
- `consts`.

So type-aware checks are cheap: the AST gives the expression shape, and the IR gives each name's type.

### Determinism (DCD0xx)

| Code | Alias | Detects | Notes |
|---|---|---|---|
| DCD001 | `wall_clock` | `time.time`, `datetime.now`, `random`, `uuid4`, `os.urandom`, env/file/network reads | Exists. Also resolve `from time import time` and import aliases; these are missed today. |
| DCD002 | `impure` | `global` / `nonlocal`, mutable defaults | Exists. Add keyword-only defaults (`kw_defaults`), which are ignored today. |
| DCD003 | `mutates_input` | `.append`, `x[k] = ...` or in-place frame ops on an argument | New |

### Numeric (DCD1xx)

| Code | Alias | Detects | Default severity |
|---|---|---|---|
| DCD101 | `int64_overflow` | int × int, `**`, `<<`, int running `+=` / `sum` where both operands are int-typed | warning; **error** when an operand is `Money(cents=True)`. Replaces today's blunt "int constant `*`" rule. |
| DCD102 | `float_money` | money held as float; float arithmetic flowing into a cents int output; `int(x * 100)` truncation | warning |
| DCD103 | `bare_round` | `round(x, n)` / `np.round` in a step (CPython and numba differ on `.xx5`) | warning; **error** on a Money output. Ship the rounding helper the money note calls for, and link it in `fix`. |
| DCD104 | `division_by_variable` | `/`, `//` or `%` by something not provably non-zero (no dominating guard, no `max(b, eps)`, not a param with `gt=0`) | warning |
| DCD105 | `int_true_division` | true division of two **int-typed** values | info. Replaces the false-positive `float_precision` rule. |
| DCD106 | `pow_exp_overflow` | `**`, `exp` or `pow` with a non-constant exponent: overflow to inf, underflow to 0 | warning |
| DCD107 | `float_equality` | `==` / `!=` on floats | warning. Exists; extend it to two float-annotated names. |
| DCD108 | `nan_propagation` | an unguarded float reaching a branch condition (NaN compares False silently); `sqrt` / `log` of a possibly non-positive value | info |
| DCD109 | `accumulator_dtype` | loop carries or `each(accumulate=)` in int64 (wraps) or float32 | warning |
| DCD110 | `unsafe_cast` | `int(float)` above 2^53, narrowing casts, `float(cents)` above 2^53 | warning |

### Structure (DCD2xx)

| Code | Alias | Detects | Default severity |
|---|---|---|---|
| DCD201 | `unreachable_branch` | a constant condition, or an arm count that doesn't fit the condition type. For trees and tables, use the step's own `check` hook. | warning |
| DCD202 | `dead_output` | a value nobody reads that isn't emitted or final | info |
| DCD203 | `shadowed_write` | a later step overwrites a value before anyone reads it (the "later write wins" trap) | warning |
| DCD204 | `unused_input` | an argument the body never uses | info |
| DCD205 | `unbounded_loop` | huge or absent `max_iterations`, or a condition that is independent of the carries | warning |
| DCD206 | `opaque_frame_step` | a frame step with unknown lineage, which blocks lineage checks, fusion and single-record scope. It absorbs `data.scope.frame_steps`. | info |

### Params (DCD3xx)

| Code | Alias | Detects | Default severity |
|---|---|---|---|
| DCD301 | `param_unconstrained` | `param()` without bounds; `Percent` without `ge=0, le=1` | info; warning for Money/Percent |
| DCD302 | `param_default_out_of_bounds` | a default that breaks its own bounds; `on_invalid` hiding bad values | warning |
| DCD303 | `param_unused` | a declared param the body never reads | warning |
| DCD304 | `params_doc_unknown_key` | a key in the params document that matches no step (a typo); needs the document | error |

### Ids (DCD4xx)

| Code | Alias | Detects | Default severity |
|---|---|---|---|
| DCD401 | `durable_ids` | missing flow or step ids | warning. Exists. |
| DCD402 | `duplicate_ids` | two definitions with one id | error |

### Data and runtime (DCD5xx, opt-in)

| Code | Alias | Needs | Detects |
|---|---|---|---|
| DCD501 | `crash_probe` | executes | runs on `testing.corpus` (zero, negative, null, extremes) and reports crashes by case. This is today's `equivalence()` without the mode comparison. |
| DCD502 | `nan_output` | data | NaN or inf in an output column |
| DCD503 | `int64_range` | data | observed column min/max pushed through DCD101 expressions by interval arithmetic, beyond 2^63 or beyond 2^53 for accumulators, with no execution |
| DCD504 | `missing_required` | data | required inputs absent or null. This is today's `preflight` data check. |
| DCD505 | `mode_divergence` | executes | interpreted, stepped and fused disagree on corpus and `.xx5` ties. It calls `testing.assert_equivalent`; the helper stays in `testing`. |
| DCD506 | `constant_output` | data | an output that is constant, or a branch arm never taken on this data |

Suggested build order:
1. DCD101–104
2. DCD202/203 and DCD301
3. DCD504/501, which retire `preflight` and `equivalence()`
4. the rest

## What moves out

- `checks.compare`, `checks.Equivalence` → `decider.diff` (see `diff.md`).
- `checks.equivalence()` → split into DCD501 and DCD505.
- `data.preflight` / `PreflightReport` → DCD304 + DCD504. The cost estimate stays with experiments.
- `data.check_scope` → renamed `plan_scope` (it plans a run, it doesn't check one), or deleted with the rest of `data.scope`.
