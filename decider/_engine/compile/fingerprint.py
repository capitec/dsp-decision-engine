from __future__ import annotations

import datetime
import enum
import fractions
import functools
import hashlib
import importlib.metadata
import os
import re
import sys
import sysconfig
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
_OWN_CODE = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + os.sep
_LIBRARY_ROOTS = tuple({os.path.join(p, "") for p in map(sysconfig.get_paths().get, ("stdlib", "purelib", "platlib"))})
_BY_VALUE = (datetime.date, datetime.time, datetime.timedelta, datetime.tzinfo, Decimal, fractions.Fraction, np.generic)


class _Walk:
    """The state of one traversal: the hash being fed, the ids already visited and what hidden state it met.

    `where` is the (name, site) of the closure variable or global being read; `found` is collected only
    when a caller asks, so the cache key and `hidden_state` are one walk and cannot disagree.
    """

    __slots__ = ("h", "seen", "strict", "found", "where")

    def __init__(self, strict: bool = False, found: list | None = None):
        self.h = hashlib.sha256()
        self.seen: set = set()
        self.strict = strict
        self.found = found
        self.where: tuple[str, str] = ("", "")


def fingerprint(fn: Callable, *, strict: bool = False) -> str:
    """A content hash of a function: what its compiled code depends on, and nothing else.

    Covers the bytecode with its constants, the argument layout, closure cell
    values and the globals it reads: functions by their own content (an
    `lru_cache` wrapper by the function it wraps; one of the standard library or
    an installed package by its name, version, code and closure, not what it
    reaches), modules by name plus the
    functions, submodules and data (`mod.LIMIT`) the function reaches through
    them, classes by `module:qualname` and their own methods and constants (an
    enum class also by its members), enum members by class and name, numbers,
    strings, dates, Decimals, regexes and tuples, lists, dicts and sets of those
    by value, numpy arrays and polars DataFrames by value. Any other object is keyed by identity, which changes
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
    w = _Walk(strict)
    _function(w, fn, root=True)
    return w.h.hexdigest()


def hidden_state(fn: Callable) -> list[tuple[str, str, str, str]]:
    """The objects `fn` reads that a config document, a diff or a kernel cannot see.

    One `(name, kind, type_name, site)` per closure variable or module global
    that holds an instance of a class of your own, or a `dict`, `list` or
    `set` that can be edited in place. `kind` is `"object"` or `"container"`.
    Modules, functions, classes, enum members, numbers, strings, dates,
    tuples and frozensets of those, arrays and frames are data, not hidden state.

    Example::

        registry = Register()

        def step(x: float) -> float:
            return registry.cap(x)

        hidden_state(step)  # [('registry', 'object', 'Register', 'module global in step.py:3')]
    """
    w = _Walk(found=[])
    _function(w, fn, root=True)
    return w.found


def cpu_target() -> tuple[str, str, str]:
    """The (LLVM triple, CPU name, CPU features) this process compiles for.

    numba's disk cache misses silently on another CPU; record this at build
    time and compare it on the serving host.

    Example::

        triple, cpu, features = cpu_target()
    """
    from numba.core.registry import cpu_target as target

    return target.target_context.codegen().magic_tuple()


def _function(w: _Walk, fn: Any, root: bool = False) -> None:
    fn = getattr(fn, "py_func", fn)
    if id(fn) in w.seen:
        w.h.update(b"<recursive>")
        return
    w.seen.add(id(fn))
    code = fn.__code__
    _code(w.h, code)
    if not root and _is_library(code.co_filename):
        # A callable of the standard library or an installed package is keyed by who and which version it is, not
        # walked: its helpers and constants are not the step's data. Its code and closure still count, as two
        # lambdas, or two closures from one factory, share a qualname; only a function it closes over (a wrapper it
        # made around the step) is the step author's to answer for. The step itself is walked wherever it is installed.
        w.h.update(f"library:{fn.__module__}.{fn.__qualname__}:{_version(fn.__module__)}".encode())
        found = w.found
        for cell in fn.__closure__ or ():
            try:
                contents = cell.cell_contents
            except ValueError:  # an empty cell
                w.h.update(b"<empty>")
                continue
            w.found = found if isinstance(contents, (types.FunctionType, Dispatcher)) else None
            _value(w, contents)
        w.found = found
        return
    names = _names(code)
    where = f"{code.co_filename}:{code.co_firstlineno}"
    outer = w.where  # a function reached through a container must not rename what the container holds next
    found = w.found
    if code.co_filename.startswith(_OWN_CODE):
        w.found = None  # the key still covers decider's helpers; what they read is not the step author's to fix
    for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError:  # an empty cell
            w.h.update(b"<empty>")
            continue
        w.where = (name, f"closure variable of {fn.__qualname__}, {where}")
        _value(w, contents, names)
    for name in sorted(names):
        if name in fn.__globals__:
            w.h.update(name.encode())
            w.where = (name, f"module global in {where}")
            _value(w, fn.__globals__[name], names)
    w.where, w.found = outer, found


@functools.lru_cache(maxsize=None)
def _is_library(filename: str) -> bool:
    if filename.startswith("<frozen"):
        return True
    # decider's own functions are walked: a step may call `decimal.round`, and a release changes them.
    return filename.startswith(_LIBRARY_ROOTS) and not filename.startswith(_OWN_CODE)


@functools.lru_cache(maxsize=None)
def _version(module: str | None) -> str:
    try:
        return importlib.metadata.version(str(module).partition(".")[0])
    except Exception:  # the standard library has no distribution, and a broken one is no reason to fail a build
        return sys.version.partition(" ")[0]


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


def _hidden(w: _Walk, kind: str, value: Any) -> None:
    from decider.steps import Step

    if w.found is None or isinstance(value, Step):
        return
    entry = (w.where[0], kind, type(value).__name__, w.where[1])
    # A variable holding a dict of objects is one finding, not one per object.
    if not any(e[0] == entry[0] and e[3] == entry[3] for e in w.found):
        w.found.append(entry)


def _by_identity(w: _Walk, value: Any) -> None:
    _hidden(w, "object", value)
    if w.strict:
        t = type(value)
        raise ValueError(f"a {t.__module__}.{t.__qualname__} object has no content key: "
                         "it is mutable or arbitrary, so it can only be keyed by identity")
    w.h.update(f"object:{type(value).__qualname__}:{id(value)}".encode())


def _qualified(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _class(w: _Walk, cls: type) -> None:
    w.h.update(f"class:{_qualified(cls)}".encode())
    if id(cls) in w.seen:
        return
    w.seen.add(id(cls))
    if issubclass(cls, enum.Enum):  # a member's value is compiled in as a constant
        _container(w, "members", [(m.name, m.value) for m in cls])
    # Its body too, or a step keeps the code it compiled against an edited method, constant or namedtuple
    # field. Methods by their own code, not what they reach; an attribute with no content key (a
    # descriptor, a registry) is left out, as keying it by identity would miss every other process.
    for name, attr in sorted(vars(cls).items()):
        attr = getattr(attr, "__func__", attr)  # a staticmethod or classmethod
        if isinstance(attr, types.FunctionType):
            w.h.update(name.encode())
            _code(w.h, attr.__code__)
            continue
        part = _Walk(True)
        part.seen = set(w.seen)  # a copy: a failed attempt must not mark what it reached as hashed
        try:
            _value(part, attr)
        except ValueError:
            continue
        w.h.update(f"{name}:{part.h.hexdigest()}".encode())


def _container(w: _Walk, kind: str, items: Any) -> None:
    # Delimited, so (1, (2,)) and ((1,), 2) differ; a container reaching itself is hashed once.
    w.seen.add(id(items))
    w.h.update(f"{kind}[{len(items)}](".encode())
    if isinstance(items, dict):
        flat = [x for pair in items.items() for x in pair]
    elif isinstance(items, (set, frozenset)):
        flat = sorted(_digest(item, w.strict) for item in items)  # sets iterate in hash-seed order
    else:
        flat = items
    for item in flat:
        _value(w, item)
    w.h.update(b")")
    w.seen.discard(id(items))


def _digest(value: Any, strict: bool) -> str:
    w = _Walk(strict)
    _value(w, value)
    return w.h.hexdigest()


def _frame(w: _Walk, df: pl.DataFrame) -> None:
    # ponytail: hashed on every call (~0.5 ms small, ~13 ms per million rows); memoize if big data globals show up.
    try:
        rows = df.hash_rows().to_numpy()
    except Exception:  # a dtype polars can't hash
        _by_identity(w, df)
        return
    w.h.update(f"frame:{df.schema}:".encode())
    w.h.update(rows.data)


def _names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names(const)
    return names


def _value(w: _Walk, value: Any, names: frozenset | set = frozenset()) -> None:
    h = w.h
    if isinstance(value, (types.FunctionType, Dispatcher)):
        _function(w, value)
    elif isinstance(value, types.ModuleType):
        h.update(f"module:{value.__name__}".encode())
        # `mod.f(x)` and `mod.K` read off the module: follow the attributes the function names, so editing one
        # changes the key. `__dict__`, not getattr: a module's `__getattr__` can run code or raise.
        key = (id(value), frozenset(names))
        if key in w.seen:
            return
        w.seen.add(key)
        contents = getattr(value, "__dict__", {})
        outer = w.where
        for name in sorted(names):
            # A callable other than a function (a ufunc, a bound method) is not data the step's code depends on.
            if name in contents and not name.startswith("__") and (
                    isinstance(contents[name], (types.FunctionType, Dispatcher, types.ModuleType, type))
                    or not callable(contents[name])):
                h.update(name.encode())
                w.where = (f"{outer[0]}.{name}", outer[1])
                _value(w, contents[name], names)
        w.where = outer
    elif isinstance(value, _Intrinsic):
        _function(w, value._defn)
    elif isinstance(value, (ir.Type, numba_types.Type)):
        h.update(f"{type(value).__name__}:{value}".encode())
    elif isinstance(value, functools._lru_cache_wrapper):
        _function(w, value.__wrapped__)  # the cache is not part of the code
    elif isinstance(value, types.BuiltinFunctionType) and isinstance(value.__self__, (types.ModuleType, type(None))):
        h.update(f"builtin:{value.__module__}.{value.__qualname__}".encode())
    elif isinstance(value, re.Pattern):
        h.update(f"regex:{value.flags}:{value.pattern!r}".encode())
    elif isinstance(value, enum.Enum):
        h.update(f"member:{_qualified(type(value))}.{value.name}".encode())
    elif isinstance(value, type):
        _class(w, value)
    elif isinstance(value, _LITERALS + _BY_VALUE):
        h.update(f"{type(value).__name__}:{value!r}".encode())
    elif (isinstance(value, tuple) or type(value) in (list, dict, set, frozenset)) and id(value) not in w.seen:
        # Exact types: a Counter or defaultdict is a mutable store, and a key that moves as it fills never hits.
        # A dict, list or set read by a step can be edited in place after the build, so it is hidden state too.
        if type(value) in (list, dict, set):
            _hidden(w, "container", value)
        _container(w, type(value).__name__, value)
    elif isinstance(value, np.ndarray) and value.dtype != object:
        # Data by value: a re-imported module's equal constant is the same step, and an in-place edit is not.
        h.update(f"ndarray:{value.dtype.str}:{value.shape}".encode())
        h.update(np.ascontiguousarray(value).data)
    elif isinstance(value, pl.DataFrame):
        _frame(w, value)
    else:
        # Anything else is compiled in as whatever object it is now: key it by identity.
        _by_identity(w, value)
