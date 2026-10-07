# Getting started

!!! note "Draft"
    Scaffold: the quickstart works, the rest is an outline.

## Install

```bash
pip install decider
```

Extras add what a task needs, for example `decider[serve-starlette]` to serve over HTTP.

## Your first pipeline

A step is a function. Its arguments are the columns it reads and its name is the column it writes. `param()` marks a policy value that can be retuned without a code change.

```python
import polars as pl
from decider import flow, param

def ratio(income: float, debt: float) -> float:
    return debt / income

def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

pipeline = flow(ratio, affordable, name="afford")

df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
pipeline.run(df)                                                     # a whole frame
pipeline.compile("fused").score({"income": 1000.0, "debt": 200.0})   # one record, compiled
```

## Outline

- Install extras and the `decider` CLI
- Run a frame, then score a single record
- Retune a param without rebuilding
- Where to go next: the user guides
