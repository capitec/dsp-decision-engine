from __future__ import annotations

import datetime
import enum
import fractions
import functools
import hashlib
import re
import types
from decimal import Decimal
from typing import Any, Callable

import numpy as np
import polars as pl
from llvmlite import ir
from numba import types as numba_types
from numba.core.dispatcher import Dispatcher
from numba.extending import _Intrinsic

_LITERALS = (int, float, complex, bool, str, bytes, type(None))
_BY_VALUE = (datetime.date, datetime.time, datetime.timedelta, Decimal, fractions.Fraction)


def fingerprint(fn: Callable, *, strict: bool = False) -> str:
    """A content hash of a function: what its compiled code depends on, and nothing else.

    Covers the bytecode with its constants, the argument layout, closure cell
    values and the globals it reads: functions by their own content (an
    `lru_cache` wrapper by the function it wraps), modules by name plus the
    functions and submodules the function reaches through them, classes by
    `module:qualname` (an enum class also by its members), enum members by
    class and name, numbers, strings, dates, Decimals, regexes and tuples,
    lists, dicts and sets of those by value, numpy arrays and polars
    DataFrames by value. Any other object is keyed by identity, which changes
    every process; `strict=True` raises instead. Leaves out the function's
    name, file and line numbers, so renaming or moving a step keeps its
    compiled code, while editing a constant, a closure value or a global it
    reads does not.

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
            contents = cell.cell_contents
        except ValueError:  # an empty cell
            h.update(b"<empty>")
            continue
        _value(h, contents, seen, strict)
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
        t = type(value)
        raise ValueError(f"a {t.__module__}.{t.__qualname__} object has no content key: "
                         "it is mutable or arbitrary, so it can only be keyed by identity")
    h.update(f"object:{type(value).__qualname__}:{id(value)}".encode())


def _qualified(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _container(h: Any, kind: str, items: Any, seen: set, strict: bool) -> None:
    # Delimited, so (1, (2,)) and ((1,), 2) differ; a container reaching itself is hashed once.
    seen.add(id(items))
    h.update(f"{kind}[{len(items)}](".encode())
    if isinstance(items, dict):
        flat = [x for pair in items.items() for x in pair]
    elif isinstance(items, (set, frozenset)):
        flat = sorted(_digest(item, strict) for item in items)  # sets iterate in hash-seed order
    else:
        flat = items
    for item in flat:
        _value(h, item, seen, strict)
    h.update(b")")
    seen.discard(id(items))


def _digest(value: Any, strict: bool) -> str:
    h = hashlib.sha256()
    _value(h, value, set(), strict)
    return h.hexdigest()


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
    elif isinstance(value, _Intrinsic):
        _function(h, value._defn, seen, strict)
    elif isinstance(value, (ir.Type, numba_types.Type)):
        h.update(f"{type(value).__name__}:{value}".encode())
    elif isinstance(value, functools._lru_cache_wrapper):
        _function(h, value.__wrapped__, seen, strict)  # the cache is not part of the code
    elif isinstance(value, types.BuiltinFunctionType) and isinstance(value.__self__, (types.ModuleType, type(None))):
        h.update(f"builtin:{value.__module__}.{value.__qualname__}".encode())
    elif isinstance(value, re.Pattern):
        h.update(f"regex:{value.flags}:{value.pattern!r}".encode())
    elif isinstance(value, enum.Enum):
        h.update(f"member:{_qualified(type(value))}.{value.name}".encode())
    elif isinstance(value, type):
        h.update(f"class:{_qualified(value)}".encode())
        if issubclass(value, enum.Enum):  # a member's value is compiled in as a constant
            _container(h, "members", [(m.name, m.value) for m in value], seen, strict)
    elif isinstance(value, _LITERALS + _BY_VALUE):
        h.update(f"{type(value).__name__}:{value!r}".encode())
    elif (isinstance(value, tuple) or type(value) in (list, dict, set, frozenset)) and id(value) not in seen:
        # Exact types: a Counter or defaultdict is a mutable store, and a key that moves as it fills never hits.
        _container(h, type(value).__name__, value, seen, strict)
    elif isinstance(value, np.ndarray) and value.dtype != object:
        # Data by value: a re-imported module's equal constant is the same step, and an in-place edit is not.
        h.update(f"ndarray:{value.dtype.str}:{value.shape}".encode())
        h.update(np.ascontiguousarray(value).data)
    elif isinstance(value, pl.DataFrame):
        _frame(h, value, strict)
    else:
        # Anything else is compiled in as whatever object it is now: key it by identity.
        _by_identity(h, value, strict)
