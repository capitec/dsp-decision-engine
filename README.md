# Decider toolkit

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The decider toolkit (package `decikit`) builds decision pipelines from plain
Python functions, run over polars frames or single records. A function's arguments are the columns it reads and its name
is the column it writes. Steps compose with `flow`, `dag`, `branch` and
`loop`; decision trees, decision tables and scorecards load from JSON as
steps. Tunable values are `param()`s, retuned per call through a params
document without rebuilding or recompiling. The same pipeline runs
interpreted, or compiled with numba (per step or fused), and can be paused and
stepped through in a debug session.

## Install

```bash
pip install decikit                        # library and CLI
pip install "decikit[serve-starlette]"     # HTTP serving with uvicorn + starlette
pip install "decikit[serve-sanic]"         # or sanic
pip install "decikit[mcp]"                 # `decikit mcp` server for agents (fastmcp)
pip install "decikit[prove]"               # symbolic proofs (z3-solver)
```

From a checkout: `uv sync`, then `uv run pytest`.

## Example

```python
import polars as pl
from decikit import FusedEngine, flow, param

def ratio(income: float, debt: float) -> float:
    return debt / income

def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

pipeline = flow(ratio, affordable, name="afford")
df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
pipeline.run(df)                                                   # income, debt, affordable
pipeline.run(df, params={"afford": {"affordable": {"limit": 0.9}}})
FusedEngine().bind(pipeline).run({"income": 1000.0, "debt": 200.0})  # one record
```

## CLI

```bash
decikit guide                 # print the getting-started guide
decikit template NAME [DIR]   # write a starter project
decikit build [VERSION]       # stage and warm a config version (fills the numba cache)
decikit serve                 # POST /invocations, GET /ping
```

Settings come from `DECIKIT_*` environment variables; see `decikit --help`.

## Claude Code plugin

This repository is a Claude Code plugin marketplace. In Claude Code:

```
/plugin marketplace add capitec/dsp-decision-engine
/plugin install decikit@decikit
```

You get the `decikit` skill (a guide to the project layout, best practices,
pipelines, checks and the config documents) and the `decikit mcp` server, which
reads pipelines, edits trees, tables, scorecards and params with validation, and
talks to the VS Code extension. The server runs through `uv` in the directory
Claude Code starts in, so start it from your project root. Setup and
troubleshooting: [setup-project.md](plugins/decikit/skills/decikit/references/setup-project.md).

## Documentation

- Start with `decikit guide` (the same text as `decikit/GUIDE.md`): concepts,
  runnable examples, project layout and common mistakes. `docs/` is the docs site.
- `docs/architecture/`: how it works (steps and IR, engine, configurable steps, serving,
  lab); published on the docs site.
- `docs/design_decisions/`: why it is built this way: decisions, experiments and known
  limitations.
- The package docstrings (`help(decikit)`) are the user documentation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `uv run pytest` before opening a
pull request.

## License

MIT. See [LICENSE](LICENSE).
