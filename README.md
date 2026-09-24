# Decider

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Decision pipelines from plain Python functions, run over polars frames or
single records. A function's arguments are the columns it reads and its name
is the column it writes. Steps compose with `flow`, `dag`, `branch` and
`loop`; decision trees, decision tables and scorecards load from JSON as
steps. Tunable values are `param()`s, retuned per call through a params
document without rebuilding or recompiling. The same pipeline runs
interpreted, or compiled with numba (per step or fused), and can be paused and
stepped through in a debug session.

## Install

```bash
pip install decider                        # library and CLI
pip install "decider[serve-starlette]"     # HTTP serving with uvicorn + starlette
pip install "decider[serve-sanic]"         # or sanic
```

From a checkout: `uv sync`, then `uv run pytest`.

## Example

```python
import polars as pl
from decider import Engine, flow, param

def ratio(income: float, debt: float) -> float:
    return debt / income

def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

pipeline = flow(ratio, affordable, name="afford")
df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
pipeline.run(df)                                                   # income, debt, affordable
pipeline.run(df, params={"afford": {"affordable": {"limit": 0.9}}})
Engine().bind(pipeline, mode="fused").score({"income": 1000.0, "debt": 200.0})   # one record
```

## CLI

```bash
decider guide                 # print the getting-started guide
decider template NAME [DIR]   # write a starter project
decider build [VERSION]       # stage and warm a config version (fills the numba cache)
decider serve                 # POST /invocations, GET /ping
```

Settings come from `DECIDER_*` environment variables; see `decider --help`.

## Documentation

- Start with `decider guide` (the same text as `decider/GUIDE.md`): concepts,
  runnable examples, project layout and common mistakes. `docs/` is outdated.
- `IR.md`: the contract, what a pipeline compiles to and how each mode runs it.
- `Design.md`: the design and package layout.
- `notes/`: decision records with their measurements; `notes/spec-amendments.md`
  amends `IR.md`.
- The package docstrings (`help(decider)`) are the user documentation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `uv run pytest` before opening a
pull request.

## License

MIT. See [LICENSE](LICENSE).
