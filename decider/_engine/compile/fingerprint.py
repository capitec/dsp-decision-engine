from __future__ import annotations

import hashlib
import types
from typing import Any, Callable

import numpy as np
import polars as pl
from numba.core.dispatcher import Dispatcher

_LITERALS = (int, float, complex, bool, str, bytes, type(None))


def fingerprint(fn: Callable, *, strict: bool = False) -> str:
    """A content hash of a function: what its compiled code depends on, and nothing else.

    Covers the bytecode with its constants, the argument layout, closure cell
    values and the globals it reads (functions by their own content, modules
    by name plus the functions and submodules the function reaches through
    them, numpy arrays and polars DataFrames by value, other objects by
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
    _function(h, fn, set(), strict)
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


def _function(h: Any, fn: Any, seen: set[int], strict: bool) -> None:
    fn = getattr(fn, "py_func", fn)
    if id(fn) in seen:
        h.update(b"<recursive>")
        return
    seen.add(id(fn))
    code = fn.__code__
    _code(h, code)
    for cell in fn.__closure__ or ():
        try:
            _value(h, cell.cell_contents, seen, strict)
        except ValueError:  # an empty cell
            h.update(b"<empty>")
    names = _names(code)
    for name in sorted(names):
        if name in fn.__globals__:
            h.update(name.encode())
            _value(h, fn.__globals__[name], seen, strict, names)


def _code(h: Any, code: types.CodeType) -> None:
    h.update(code.co_code)
    h.update(repr((code.co_argcount, code.co_kwonlyargcount, code.co_flags,
                   code.co_varnames, code.co_names, code.co_freevars)).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _code(h, const)
        else:
            # A set's repr order follows its elements' hash, which the process's
            # hash seed moves, so a membership test would fingerprint differently per process.
            const = sorted(const, key=repr) if isinstance(const, (set, frozenset)) else const
            h.update(f"{type(const).__name__}:{const!r}".encode())


def _by_identity(h: Any, value: Any, strict: bool) -> None:
    if strict:
        raise ValueError(f"{type(value).__qualname__} can only be keyed by identity")
    h.update(f"object:{type(value).__qualname__}:{id(value)}".encode())


def _frame(h: Any, df: pl.DataFrame, strict: bool) -> None:
    # ponytail: hashed on every call (~0.5 ms small, ~13 ms per million rows); memoize if big data globals show up.
    try:
        rows = df.hash_rows().to_numpy()
    except Exception:  # a dtype polars can't hash
        _by_identity(h, df, strict)
        return
    h.update(f"frame:{df.schema}:".encode())
    h.update(rows.data)


def _names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names(const)
    return names


def _value(h: Any, value: Any, seen: set, strict: bool, names: frozenset | set = frozenset()) -> None:
    if isinstance(value, (types.FunctionType, Dispatcher)):
        _function(h, value, seen, strict)
    elif isinstance(value, types.ModuleType):
        h.update(f"module:{value.__name__}".encode())
        # `mod.f(x)` reads `f` off the module: follow the attributes the function names, so editing `f` changes the key.
        key = (id(value), frozenset(names))
        if key in seen:
            return
        seen.add(key)
        for name in sorted(names):
            attr = value.__dict__.get(name)
            if isinstance(attr, (types.FunctionType, Dispatcher, types.ModuleType)):
                h.update(name.encode())
                _value(h, attr, seen, strict, names)
    elif isinstance(value, _LITERALS):
        h.update(f"{type(value).__name__}:{value!r}".encode())
    elif isinstance(value, tuple):
        for item in value:
            _value(h, item, seen, strict)
    elif isinstance(value, np.ndarray) and value.dtype != object:
        # Data by value: a re-imported module's equal constant is the same step, and an in-place edit is not.
        h.update(f"ndarray:{value.dtype.str}:{value.shape}".encode())
        h.update(np.ascontiguousarray(value).data)
    elif isinstance(value, pl.DataFrame):
        _frame(h, value, strict)
    else:
        # Anything else is compiled in as whatever object it is now: key it by identity.
        _by_identity(h, value, strict)
