# Best practices

Use this when you decide where a number or rule lives, review a change before
it ships, or catch a mistake before the user does. Each rule closes a mistake
agents make in decikit projects.

## Where each thing lives

| Thing | Lives in | Why |
|---|---|---|
| Request data (amount, dates, ids) | a plain function argument | a `param()` ignores the request |
| Policy that is the same for every request (limit, cut-off, weight) | `param(default, ge=, le=)` in Python, value in `params.json` | retuned per version with no code change or recompile |
| Tables, bands, thresholds, rate cards, rule sets | a config document in `configs/<version>/` (tree, rule set, decision table, scorecard) | reviewable as data, validated before it is saved, versioned |
| Calculation per record | a plain function | the engine compiles it |
| Today's date, randomness, ids | the request (`applied_on`) | `date.today()`, `hash()` and `uuid4()` make the same request decide differently |

Use `TreeConfig`, `DecisionTableConfig` and `ScorecardConfig` for trees, rule
loops and points tables; hand-written walkers and if-chains lose validation,
tracing and the equivalence guarantees. Section "Which construct for which
need" in `uv run decikit guide` maps each need to its construct.

A step writes one value. Write one step per output (or
`@step(outputs=(...))` with a `tuple` return); a step returning a `dict`
breaks the wiring.

## Params

- Give every `param()` bounds when a wrong value is possible:
  `param(0.4, ge=0.0, le=1.0)`. The `param_unconstrained` check flags the rest.
- Name a param for the policy it expresses (`month_end_limit`), so the params
  document reads as the policy.
- Retune by editing the params document of a new config version, not the
  default in code. `pipeline.parameters().defaults()` prints the whole document.
- A number inside a config document becomes a param with
  `{"param": "name", "default": x}`.

## Durable ids

`decikit ids [PATH]` writes an opaque `id="0123abcdef45"` into every named flow
and `@step` that lacks one, and changes no other byte. Experiments, diffs and
traces match steps by this id, so a rename or reorder keeps its history.

```bash
uv run decikit ids credit_risk --check     # preview, writes nothing
uv run decikit ids credit_risk             # needs every affected file git-tracked and a clean tree
```

Commit the ids in their own commit. `generate_ids(path, check=True)` previews
the same change through MCP. A plain function listed in `flow(...)` carries no
id; decorate it with `@step(id=...)` to give it one.

## Checks

`checks.standard` runs wall-clock reads, shared-state writes, numeric risk,
missing ids and unknown suppressions. The project's policy is `checks.yaml`
beside `configs/`:

```yaml
extends: standard
checks:
  - {type: durable_ids, severity: error}
fail_on: error
```

`uv run decikit check credit_risk.pipeline:build` exits 1 when a finding reaches
`fail_on`. Fix the finding; suppress only when the code is right and say why,
with `suppress_checks=["durable_ids"]` on the step. testing-and-checks.md covers
reading a report.

## Exact-answer tests

Assert the answer of a worked example computed by hand, at the boundary where
the rule changes:

```python
(1000.0, 400.0, date(2026, 1, 10), True)    # ratio 0.40 meets the 0.4 limit
(1000.0, 401.0, date(2026, 1, 10), False)   # one rand over
```

`assert out >= 0` and `assert out in (1, 2, 3)` pass for a wrong rule. Test
through decikit (`Engine().bind(build()).run(record, params)` or the
handler), never by calling the functions, so the test proves the pipeline wires,
builds and serves. testing-and-checks.md has the full pattern.

## Versioning and replay

Config versions are immutable. A new policy is a new version
(`store.create_version`); the old version stays so a stored decision replays.

```python
from datetime import date
from decikit import Engine
from decikit.config import JsonFileStore
from credit_risk.pipeline import build

stored = {"income": 1000.0, "debt": 350.0, "applied_on": date(2026, 1, 10)}   # the request as logged
params = JsonFileStore(basepath="configs").read("0.0.0").config["params"]      # the version it ran under
Engine().bind(build(), mode="fused").run(stored, params)["approved"]           # True
```

Replay against `store.read(version)`, never whatever `configs/` holds now. A
pipeline with config documents also needs those documents from the same
version.

## Experiments before a policy change

Run the proposed params as a scenario over real records and read how many
decisions change before releasing a version (testing-and-checks.md). Use
`decikit diff main HEAD credit_risk.pipeline:build --data loans.parquet` to see
which rows a code change moves.

## Before saying it works

1. The changed documents validate (the editing tools report `problems`).
2. The worked examples pass under `uv run pytest`.
3. `uv run decikit check <entry>` has no new finding at the gate.
4. `uv run decikit build` succeeds and scoring `sample_request.json` over
   `decikit serve` returns the expected answer.

Two more slips: `RequestHandler` imports as
`from decikit.serving import RequestHandler`, and CLI options come from
`decikit --help` (settings are `DECIKIT_*` variables or `.env`, so
`--code-path` does not exist).
