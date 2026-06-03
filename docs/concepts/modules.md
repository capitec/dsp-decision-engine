# Modules

A **module** is the fundamental unit of computation in Decider. It takes a Polars `DataFrame` as input and returns a `DataFrame` with new or transformed columns.

## Defining a module from functions

Use `generate_from_functions` to turn plain Python functions into a module. Each function becomes a computed column; its parameter names map to input columns.

```python
import polars as pl
from decider import generate_from_functions

def score(income: float, debt: float) -> float:
    return income / (debt + 1)

def risk_band(score: float) -> str:
    if score > 5:
        return "low"
    elif score > 2:
        return "medium"
    return "high"

Scorer = generate_from_functions(score, risk_band, name="Scorer")
```

## Injecting configuration

Add a `config` parameter typed with a Pydantic model to inject versioned parameters:

```python
from pydantic import BaseModel

class ScorerConfig(BaseModel):
    threshold: float = 5.0

def risk_band(score: float, config: ScorerConfig) -> str:
    return "low" if score > config.threshold else "high"
```

## Module types

| Type | Description |
|---|---|
| `ExpressionModule` | Parallel column computation from functions |
| `SequentialModule` | Ordered chain of modules (output feeds next) |
| `JoinModule` | Merges two module outputs side-by-side |
| `GraphModule` | JSON-serialisable wrapper for any module |

## Running a module

```python
df = pl.DataFrame({"income": [50000, 30000], "debt": [10000, 25000]})
result = Scorer.load({}).run(df)
```
