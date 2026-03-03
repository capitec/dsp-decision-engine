"""
This was created as a way to import modules without side-efects of adding values to path.
It comes with an issue that if an imported module has relative lazy imports they will fail.
E.G. The below will fail because helper is relative to the module and at the time of use it will not be in path.
```python
def func(v1: pl.Expr, v2: pl.Expr) -> pl.Expr:
    from helper import mul
    return mul(v1, v2)
```

Yet the following will work. I think its a fair tradeoff to make for the purity of the function.
```python
from helper import mul
def func(v1: pl.Expr, v2: pl.Expr) -> pl.Expr:
    return mul(v1, v2)
```
"""
import typing as t
import sys
from contextlib import contextmanager
import importlib
from pathlib import Path

@contextmanager
def add_to_path(path: t.Optional[str]):
    if path is None:
        yield
        return
    path = str(Path(path).resolve())
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)

def import_module_with_path(module_name: str, path: str):
    with add_to_path(path):
        return importlib.import_module(module_name)
