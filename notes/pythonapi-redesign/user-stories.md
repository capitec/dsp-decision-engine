# Developer user stories

What people actually do with decider, grouped by role. Each story gives a
rating for today, the code it takes today, and the code it should take after
the redesign. The "today" code was checked against the GUIDE, the tests and
the `__init__` files. The "after" code is proposed.

Ratings:
- **clean**: keep as is.
- **awkward**: works, but needs deep imports, two packages or hidden knowledge.
- **missing**: there is no way to do it, or the result is wrong.

Roles:

| Role | Who |
|---|---|
| **Author** | writes pipelines and steps |
| **Analyst** | tunes params, runs experiments over data |
| **Engineer** | embeds, serves, runs CI |
| **Extender** | writes custom steps, checks, metrics, plots, providers |
| **Reviewer** | checks quality and compares versions |
| **Integrator** | builds VS Code / MCP on top |

---

## Author

### A1. Write a pipeline from functions and run it — clean
```python
from decider import flow, param
def ratio(income: float, debt: float) -> float: return debt / income
def approved(ratio: float, limit: float = param(0.4)) -> bool: return ratio <= limit
pipeline = flow(ratio, approved, name="credit")
pipeline.run(df)
```
No change.

### A2. Use a tree / table / scorecard document — awkward
Today:
```python
from decider.steps.trees import TreeConfig            # its __all__ has 64 names; you need 1
from decider.steps.tables import DecisionTableConfig
from decider.steps.scorecard import ScorecardConfig
```
After:
```python
from decider import flow, TreeConfig, DecisionTableConfig, ScorecardConfig   # lazy: no numba on `import decider`
```
Each subpackage's `__all__` shrinks to its `*Config` class. The document grammar classes move to `decider.steps.trees.schema` and friends, for people who build documents in Python.

### A3. Define the flow in YAML, with the steps in Python — awkward
Today:
```python
from decider.flowfile import load_flow     # not reachable from `import decider`
pipeline = load_flow("flow.yaml", components="myapp.components")
```
After:
```python
from decider import load_flow
```
Flow files can't express `dag`, `each` or `optimise` today (see `extensibility-and-coupling.md`).

### A4. Know what a pipeline looks like (nodes, edges, value slots) — awkward
Today:
```python
from decider.contract import describe
```
After:
```python
from decider import describe
```
`decider.contract` stays as the home of the reference types.

### A5. Catch decider's errors — awkward
Today:
```python
from decider.exceptions import DeciderError, ParamsError, MissingInputError, WiringError
```
`decider.exceptions` also holds HTTP errors (415/400/406) and Arrow internals. `ParamsError` and `ExprError` are each exported from two places.

After:
```python
from decider import DeciderError
from decider.exceptions import ParamsError, MissingInputError, WiringError
```
After the change, `decider.exceptions` holds only the errors a user handles. HTTP errors move to `serving` and Arrow errors to `_engine`.

### A6. Annotate units and nested data — clean
```python
from decider import Money, Percent, Duration, Columnar, Struct
```
The modules behind these are misnamed: `fields.py` holds display units, and `types.py` shadows the stdlib and mixes 4 public markers with about 15 internal helpers. The top-level names don't change.

### A7. Add durable ids to my source — awkward
Today: the `decider ids` command (fine), or `from decider.ids import add_ids`. `ids.py` is a git- and AST-rewriting dev tool, and its name says nothing.

After: the CLI is unchanged, and the module becomes private (`decider/_tools/ids.py`). Only the CLI and MCP call it.

---

## Analyst

### B1. See the tunable params, edit them, rerun — clean
```python
params = pipeline.parameters().defaults()
params["credit"]["approved"]["limit"] = 0.3
pipeline.run(df, params=params)
```

### B2. Validate a params document without running — missing
`validate_node`, `check_namespaces` and `Validation` exist in `engine.params`, but they are internal and not in the GUIDE.

After:
```python
problems = pipeline.parameters().validate(params)   # Validation; .raise_for_errors()
```

### B3. Load a dataset — awkward
Today:
```python
from decider.data import load
df = load("loans.parquet").frame
```
`load` returns a wrapper shaped like a run manifest, and it accepts `job=` (a protocol type). The lazy reader `scan` lives in `decider.data.formats` and leaks out as `decider.experiments.scan`.

After:
```python
from decider.data import read, scan
df = read("loans.parquet")     # pl.DataFrame
lf = scan("loans.parquet")     # pl.LazyFrame
```
`LoadedData`, which adds a fingerprint and a key, is still returned by `dataset(...).load()` for run recording.

### B4. Use the project's dataset catalog — clean-ish
Today:
```python
from decider.data import datasets, dataset
datasets()["loans"].load(); dataset("loans").scan()
```
The same namespace also holds `dataset_documents`, `datasets_folder` and `project_root`, which are bridge helpers. There is no way to register a dataset in a notebook.

After: the bridge helpers become private, and `datasets()` returns a mapping you can add to in memory:
```python
from decider.data import datasets, FileProvider
datasets().add("tmp", FileProvider(path="x.csv"))
```

### B5. Compare baseline params against a scenario — clean
```python
from decider.experiments import Experiment
from decider.experiments.metrics import Divergence      # today: decider.experiments (81-name flat namespace)
from decider.experiments.plots import Changes
result = (Experiment("drift", pipeline, df)
          .scenario("limit_0.3", params={"credit": {"approved": {"limit": 0.3}}})
          .metric(Divergence(column="approved")).plot(Changes(metric="divergence")).run())
result.show()
```
The only change is that the imports come from subnamespaces. `Experiment.__init__` also takes 11 arguments, 5 of which are protocol-only (`entry`, `decider`, `input_path`, `input_dataset`, `determinism`). Those move to `Experiment.from_def(def_)`.

### B6. Compare against another git revision — awkward, with a bug
Today:
```python
Experiment("x", pipeline, df, entry="pkg.mod:build").scenario("prev", revision="HEAD^")...
```
Leave out `entry=` and the run fails inside the worker subprocess. `validate` doesn't catch it.

After: `entry` is inferred from the pipeline's source location, and `validate` reports it if inference fails. For a quick check without an experiment, see R6.

### B7. Debug one record step by step — clean to use, awkward to type
Today:
```python
s = pipeline.session(df)            # typed -> Any
s.break_at("credit/approved"); s.resume(); s.set("ratio", 0.1); s.resume(); s.output()
```
`Session` lives in `decider.engine.debug` next to 29 wire-protocol names.

After:
```python
from decider.debug import Session   # Step.session -> Session
from decider.debug import Paused, NodeFinished    # events read from s.events
```
The wire models (`Command`, `COMMAND`, `EVENT`, ...) become private.

### B8. See why one record got its answer (trace) — awkward
Today:
```python
from decider.engine.trace import TraceSink, Kind
```
That package also exports `pack`, `unpack`, `bitcast`, `decode` and `SCHEMA_VERSION`. `trace=` exists on `Executable.run`/`score` but not on `Step.run`.

After:
```python
from decider.trace import TraceSink, Kind
pipeline.run(df, trace=sink)
```

---

## Engineer

### E1. Bind once and score single records fast — awkward
Today:
```python
from decider import Engine
exe = Engine().bind(pipeline, mode="fused")   # returns Executable: only importable from decider.engine.run
exe.score({"income": 1000.0, "debt": 200.0}, params)
```
After:
```python
from decider import Executable
exe: Executable = pipeline.bind("fused")      # Engine(strict_compile=True).bind(...) still there
```

### E2. Serve a project over HTTP — clean (CLI), awkward (code)
`decider template / build / serve` are fine as they are. In code:
```python
from decider.serving import RequestHandler
```
The awkward parts are:
- `RequestHandler` exposes SageMaker container names: `stage`, `activate`, `process_fn`, `init_fn`, `input_fn` and `output_fn`.
- `decider.settings` builds a global `DeciderSettings()` at import time, so `DECIDER_*` env vars and `.env` change library behaviour without you asking.
- "A thing that gives a Step" has three spellings: a `Step`, a `"module:build"` string and a `.yaml` path.

After: settings become lazy (`decider._settings.get_settings()`, read on first use, not at import). The handler surface is left alone for now; it is fixed by the SageMaker contract.

### E3. Run checks in CI — missing
Today you write Python: `check.run(...)` then `assert report.ok`.

After:
```bash
decider check credit.pipeline:build --format sarif > checks.sarif      # policy and gate from the project's checks.yaml
```

### E4. Run an experiment.json in CI — awkward, with a bug
Today:
```python
Experiment.load("experiments/drift/experiment.json").run(out_dir="results/drift")
# or: run(ExperimentDef.model_validate_json(...))   <- resolves input.path against the cwd, not the file
```
There are three entry points (`run`, `run_experiment`, `Experiment.load().run()`). No CLI command exists, and nothing gives an exit code for PARTIAL / NON_REPRODUCIBLE.

After:
```bash
decider experiment run experiments/drift/experiment.json --out results/drift   # exit 1 unless COMPLETED
```
```python
from decider.experiments import run
run("experiments/drift/experiment.json", out_dir=...)   # path or ExperimentDef; paths relative to the file
```

### E5. Track and cancel a long job — awkward
Today:
```python
from decider.lifecycle import JobHandle
job = JobHandle(job_id="j1", kind="experiment_run")      # user invents id and kind
Experiment(...).run(job=job)                              # job.cancel() from another thread
```
There are two progress mechanisms (`Callback.on_progress` and `JobHandle.progress`).

After:
```python
handle = experiment.start()                               # background; .status .progress .cancel() .result()
```
`JobHandle` becomes an integration detail in `decider.contract`.

### E6. Unit-test that modes agree — clean
```python
from decider.testing import assert_equivalent, corpus, no_recompile
```
Two gaps:
- `corpus` lacks the `.xx5` ties and large-principal rows that `money-rounding-and-overflow.md` promises.
- `corpus` raises `ValueError` on a flow with no inputs.

---

## Extender

### X1. Write a config-driven step type — clean (simple), missing (anything else)
The simple case:
```python
from decider import ConfigurableStep, Value
class Scaled(ConfigurableStep):
    type: Literal["scaled"] = "scaled"
    column: str
    factor: Value[float] = 1.0
    def to_ir(self, ctx): return ctx.call(self, scale, inputs={"x": self.column}, values={"factor": self.factor})
```
Anything past `ctx.call` (row kernels, packed tables) needs `decider.engine.ir.decls` and `.nodes`. No docs cover it. The test `tests/test_imports.py` pins them as public anyway.

After:
```python
from decider.ir import Input, Output, ParamDecl, NullPolicy, CallNode, IRContext
```
This is one documented kit, so the rest of the engine can go private.

### X2. Ship a step type in my own package — missing
Today a document can name the type only if something already imported its module, or if the document spells it `"mypkg.steps:Scaled"`.

After, using entry points:
```toml
[project.entry-points."decider.steps"]
scaled = "mypkg.steps:Scaled"
```
`ConfigurableStep.load({"type": "scaled", ...})` then works without an import. The same holds for every registry root, and checks get their own group (`decider.checks`).

### X3. Write a custom check — awkward
Today:
```python
from decider.check import Finding, Severity
def my_check(pipeline): return [Finding(severity=Severity.WARNING, message="...")]
check.run(pipeline, suites=[checks.default_suite, (my_check,)])   # nested sequences, splat the default
```
It can't be named from config, can't be configured or suppressed, and has no stable code.

After (details in `checks.md`):
```python
from decider.checks import Check
class NoFloatMoney(Check):
    type: Literal["no_float_money"] = "no_float_money"
    code = "ACME001"
    allow: list[str] = []
    def run(self, ctx): ...
checks.run(pipeline, [checks.standard, NoFloatMoney(allow=["fee"])])
```
In config this is `{"type": "no_float_money", "allow": ["fee"]}`.

### X4. Write a custom metric — clean
```python
from decider.experiments.metrics import Metric, Summary, Batch    # today: decider.experiments
```
It is registry-based, so `experiment.json` names it by `type`. One gap: the GUIDE doesn't say a custom metric must be imported first, or spelled `"module:Class"`. Entry points (X2) fix this.

### X5. Write a custom plot — awkward
`altair()` lives in `decider.experiments.plots.base`, and the plot docstring uses it without importing it.

After:
```python
from decider.experiments.plots import Plot, require_altair
```

### X6. Write a dataset provider — clean
```python
from decider.data import DatasetProvider
```
It is registry-based and named in `datasets/config.json`. No change.

---

## Reviewer

### R1. Run the standard checks — awkward
Today:
```python
from decider import check, checks        # two packages for one job
report = check.run(pipeline)
```
After:
```python
from decider import checks
report = checks.run(pipeline)            # checks.standard
print(report.to_text()); report.ok; report.counts
```

### R2. Get money overflow / rounding / division warnings — missing
Here is a verified example:
```python
def inst(cents: Annotated[int, Money(cents=True)], rate: float = param(0.1)) -> Annotated[int, Money(cents=True)]:
    return round(cents * rate / 12)   # flagged as int true division (false positive); bare round() not flagged
def ratio(a: float, b: float) -> float: return a / b      # not flagged
def sq(cents: int) -> int: return cents ** 2              # flagged (correct)
```
After: `DCD103 bare_round` fires (an error, because the output is Money), `DCD104 division_by_variable` fires on `ratio`, and `DCD101 int64_overflow` fires on `sq`. `inst` raises no false positive. The full catalogue is in `checks.md`.

### R3. Accept a known-safe finding — missing
After:
```python
@step(suppress_checks=["int64_overflow"])     # or the code, "DCD101"; also on flow/dag/branch/... and flow files
def capped(cents: int) -> int: return cents * 1000
```
A suppressed finding stays in the report and is marked suppressed. An unknown name is reported as DCD403. See `checks.md`.

### R4. Run data-dependent checks against my dataset — awkward
Today, `decider.data.preflight` returns free-text `errors: tuple[str]` with no location. It is a third report shape, separate from checks.

After:
```python
checks.run(pipeline, data=frame, params=doc)
```
Data checks emit normal `Finding`s. Without `data=`, they are listed under `report.skipped`.

### R5. See what changed between two versions, without running — missing (and today's answer can be wrong)
Today, `checks.compare(a, b)` returns `equivalent=True` for flows that differ only in a constant.

After:
```python
from decider import diff
s = diff.structure(old, new)
s.added, s.removed, s.renamed, s.changed_steps, s.params, s.identical
```
Steps are matched by durable id, so a rename shows up as a rename.

### R6. See which rows and columns change, and by how much — awkward
Today this only works inside an experiment, with one `Divergence` metric per column.

After:
```python
o = diff.outputs(old, new, frame, key="application_id")
o.rows_changed; o.columns["offer"].max_abs; o.table()
diff.revisions("main", "HEAD", entry="credit.pipeline:build").outputs(frame)
```
On the command line: `decider diff main HEAD credit.pipeline:build --data loans`.

### R7. Prove a refactor changed nothing — missing
After:
```python
diff.prove(old, new)
```
It returns `Proof(tier="ir_hash", identical=True)` when the IR hashes match. Otherwise it returns `identical=None` with the evidence. It never returns `True` from sampled rows. See `diff.md`.

---

## Integrator

### I1. Launch the editor's Python side — awkward
Today: `python -m decider.debug_bridge --fd 3`. Replies are ad-hoc dicts, there is no version handshake, and `protocol.ts` is written by hand.

After: the module path is frozen. A `{"cmd": "hello"}` request returns `{protocol, contract_version, capabilities}`. Reply models are declared in `decider.contract`, and `protocol.ts` is generated from them, with a test that fails on drift.

### I2. One flow identity — awkward
Today there are two identities:
- `describe()` gives `FlowRef` / `StepRef` with durable ids.
- The bridge's panel payload (`contract/describing.py::node_json`) gives a path plus an id, with different field names.

After: `describing.py` moves into the bridge and carries the contract's `flow_id` / `step_id`.

### I3. Run the MCP server — clean (CLI)
`decider mcp [--raw]` is fine. The library API over-exports `editor_dir` and `ws_hash`.

Two other issues:
- `fastmcp` is a hard dependency; it moves to an `mcp` extra.
- The editor-bound tools (highlight, reveal, selection) have no extension-side server yet. Either hide them until they do, or ship the server.
