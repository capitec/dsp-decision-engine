# Decider

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Decider** is a Python framework for building, serving, and inspecting decision pipelines as versioned, deployable micro-services. Define pipelines from plain Python functions, compose them with `|` and `&`, save them as versioned JSON configs, and serve them over HTTP — all with a single consistent API.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Concepts](#concepts)
- [Usage Examples](#usage-examples)
- [CLI](#cli)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Decider is built around a few core ideas:

- **Functions as nodes** — plain Python functions that accept and return `polars.Expr` are wired into executable DAGs automatically, with no decorators or registries required.
- **Composable pipelines** — modules chain with `|` (sequential) or merge with `&` (parallel union), making it easy to build complex pipelines from simple parts.
- **Versioned configs** — every pipeline is serialisable to JSON. Configs are versioned, loadable by the server at startup, and hot-swappable without redeployment.
- **Pluggable extensions** — new module types are registered into a discriminated union at runtime, so the server always knows how to reconstruct any module from its config.

## Installation

```bash
# Core library and CLI
pip install decider

# With serving dependencies
pip install "decider[serve-starlette]"   # uvicorn + starlette
pip install "decider[serve-sanic]"       # sanic

# With the interactive graph visualiser
pip install "decider[visualise]"

# Everything
pip install "decider[all]"
```

**With uv (recommended):**

```bash
git clone https://github.com/capitec/dsp-decision-engine.git
cd dsp-decision-engine
uv sync --all-extras
```

## Concepts

### Modules

A module is the basic unit of computation. The simplest way to create one is `generate_from_functions`, which turns plain functions into an executable module:

- **Function name → output column** — `def dti_ratio(...)` produces a `dti_ratio` column.
- **Parameter name → input column or sibling output** — parameters are resolved from the input DataFrame or from the output of another function in the same module.
- **`config` parameter → Pydantic model injection** — declare `config: MyConfig` and the config fields are promoted onto the module itself.

### Pipelines

Modules compose with two operators:

- `|` — **sequential**: `step_a | step_b | step_c` passes each step's output as the next step's input.
- `&` — **union**: `module_a & module_b` merges both modules into a single compilation pass, computing all columns in one frame scan.

### Versioned Configs

Every module can be saved to a versioned JSON config and reconstructed from it:

```python
await module.asave("main", config_manager)
await config_manager.save_version(overwrite=True)
```

The server reads the latest version on startup and watches for updates.

### Extensions

Custom module types are registered with `register_graph_module` and auto-discovered by `initialize_decider` from an `extension_path` directory. This keeps the core library small while allowing domain-specific modules to be developed and shipped independently.

## Usage Examples

### Your first module

```python
import polars as pl
from decider.modules.functional import generate_from_functions

def dti_ratio(debt: pl.Expr, income: pl.Expr) -> pl.Expr:
    return debt / income

def credit_score(dti_ratio: pl.Expr) -> pl.Expr:
    return pl.lit(800) - dti_ratio * 200

Scorer = generate_from_functions("credit_scorer", dti_ratio, credit_score)
scorer = Scorer(name="scorer")

df = pl.DataFrame({"debt": [25_000.0], "income": [50_000.0]})
result = scorer({"input": df})
# shape: (1, 4) — debt, income, dti_ratio, credit_score
```

### Config injection

```python
from pydantic import BaseModel

class ScorerConfig(BaseModel):
    dti_weight: float = 200.0
    score_base: float = 800.0

def credit_score(dti_ratio: pl.Expr, config: ScorerConfig) -> pl.Expr:
    return pl.lit(config.score_base) - dti_ratio * config.dti_weight

Scorer = generate_from_functions("credit_scorer", dti_ratio, credit_score)
scorer = Scorer(name="scorer", dti_weight=150.0)  # config fields on the module
```

### Sequential pipeline

```python
features = FeatureModule(name="features")
scorer   = ScorerModule(name="scorer")
flags    = FlagModule(name="flags")

pipeline = features | scorer | flags
result = pipeline({"input": df})
```

### Join then score

```python
from decider.modules.primitives.join import JoinModule

join = JoinModule(name="enrich", left="transactions", right="users", on="user_id", how="left")
pipeline = join | scorer

result = pipeline({"transactions": txns_df, "users": users_df})
```

### Save and load

```python
import asyncio
from decider.config.file import JsonFileConfigManager
from decider.modules import GraphModule

mgr = JsonFileConfigManager(basepath="./configs")
asyncio.run(scorer.asave("main", mgr))
asyncio.run(mgr.save_version(overwrite=True))

# Reconstruct from disk
fresh = JsonFileConfigManager(basepath="./configs")
loaded = asyncio.run(fresh.get_latest())
module = GraphModule.model_validate(loaded.config["main"]).root
```

## CLI

```bash
# Scaffold a new module
decider template module CreditScorer

# Scaffold a module into a shareable package
decider template module CreditScorer --package mylib

# Scaffold a new project
decider template project fraud_detection

# Start a server
decider serve                            # starlette on :8080
decider serve --engine sanic --workers 4
decider serve --port 9000 --reload

# Launch the interactive graph visualiser
decider visualise --project-dir projects/loan_scoring
```

### Jupyter magic

```python
%load_ext decider.magics
```

```python
%%module CreditScorer

class CreditScorerConfig(BaseModel):
    weight: float = 200.0

def dti_ratio(debt: pl.Expr, income: pl.Expr) -> pl.Expr:
    return debt / income

def credit_score(dti_ratio: pl.Expr, config: CreditScorerConfig) -> pl.Expr:
    return pl.lit(800) - dti_ratio * config.weight
```

This writes `decider_extensions/credit_scorer/__init__.py`, reloads it, and injects `CreditScorer` into the notebook namespace. Add `--package mylib` to write into a proper uv src-layout package instead.

## Contributing

We welcome contributions to Decider! Please refer to our [Contributing Guide](CONTRIBUTING.md) for full details.

- **Fork** the repository and create a branch from `main`.
- **Install dependencies** with `uv sync --all-extras`.
- **Run tests** with `uv run pytest`.
- **Submit a Pull Request** with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for your interest in Decider! We look forward to your contributions and feedback.
