from __future__ import annotations

import hashlib
import types
from typing import Any, Callable

import numpy as np
import polars as pl
from numba.core.dispatcher import Dispatcher

_LITERALS = (int, float, complex, bool, str, bytes, type(None))


def fingerprint(fn: Callable) -> str:
    """A content hash of a function: what its compiled code depends on, and nothing else.

    Covers the bytecode with its constants, the argument layout, closure cell
    values and the globals it reads (functions by their own content, modules
    by name, numpy arrays and polars DataFrames by value, other objects by
    identity). Leaves out the function's name, file and line numbers, so
    renaming or moving a step keeps its compiled code, while editing a
    constant, a closure value or a global it reads does not.

    Example::

        def f(x: float) -> float:
            return x * 2.0

        def g(x: float) -> float:
            return x * 2.0

        fingerprint(f) == fingerprint(g)  # True
    """
    h = hashlib.sha256()
    _function(h, fn, set())
    return h.hexdigest()


def cpu_target() -> tuple[str, str, str]:
    """The (LLVM triple, CPU name, CPU features) this process compiles for.

    numba's disk cache misses silently on another CPU; record this at build
    time and compare it on the serving host.

    Example::

        triple, cpu, features = cpu_target()
    """
    from numba.core.registry import cpu_target as target

    return target.target_context.codegen().magic_tuple()


def _function(h: Any, fn: Any, seen: set[int]) -> None:
    fn = getattr(fn, "py_func", fn)
    if id(fn) in seen:
        h.update(b"<recursive>")
        return
    seen.add(id(fn))
    code = fn.__code__
    _code(h, code)
    for cell in fn.__closure__ or ():
        try:
            _value(h, cell.cell_contents, seen)
        except ValueError:  # an empty cell
            h.update(b"<empty>")
    for name in sorted(_names(code)):
        if name in fn.__globals__:
            h.update(name.encode())
            _value(h, fn.__globals__[name], seen)


def _code(h: Any, code: types.CodeType) -> None:
    h.update(code.co_code)
    h.update(repr((code.co_argcount, code.co_kwonlyargcount, code.co_flags,
                   code.co_varnames, code.co_names, code.co_freevars)).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _code(h, const)
        else:
            h.update(f"{type(const).__name__}:{const!r}".encode())


def _frame(h: Any, df: pl.DataFrame) -> None:
    # ponytail: hashed on every call (~0.5 ms small, ~13 ms per million rows); memoize if big data globals show up.
    try:
        rows = df.hash_rows().to_numpy()
    except Exception:  # a dtype polars can't hash
        h.update(f"object:DataFrame:{id(df)}".encode())
        return
    h.update(f"frame:{df.schema}:".encode())
    h.update(rows.data)


def _names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names(const)
    return names


def _value(h: Any, value: Any, seen: set[int]) -> None:
    if isinstance(value, (types.FunctionType, Dispatcher)):
        _function(h, value, seen)
    elif isinstance(value, types.ModuleType):
        h.update(f"module:{value.__name__}".encode())
    elif isinstance(value, _LITERALS):
        h.update(f"{type(value).__name__}:{value!r}".encode())
    elif isinstance(value, tuple):
        for item in value:
            _value(h, item, seen)
    elif isinstance(value, np.ndarray) and value.dtype != object:
        # Data by value: a re-imported module's equal constant is the same step, and an in-place edit is not.
        h.update(f"ndarray:{value.dtype.str}:{value.shape}".encode())
        h.update(np.ascontiguousarray(value).data)
    elif isinstance(value, pl.DataFrame):
        _frame(h, value)
    else:
        # Anything else is compiled in as whatever object it is now: key it by identity.
        h.update(f"object:{type(value).__qualname__}:{id(value)}".encode())
