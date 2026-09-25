from __future__ import annotations

import hashlib
import inspect
import os
import types as py_types
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numba import from_dtype, njit, typeof
from numba.core import types
from numba.core.caching import FunctionCache
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError, UnsupportedBytecodeError

from decider.engine.compile import cpython  # registers CPython-compatible round and **
from decider.engine.compile.fingerprint import fingerprint
from decider.engine.ir.decls import KIND_DTYPES, FeatureKind, Input, NullPolicy, base_annotation, feature_kind
from decider.engine.ir.nodes import CallNode
from decider.engine.params import NodeParams
from decider.steps.helpers import helper_signatures, is_python_only
from decider.types import is_raw, raw_base

# UnsupportedBytecodeError (e.g. an `import` inside a step) isn't a NumbaError.
FALLBACK_ERRORS = (NumbaError, UnsupportedBytecodeError)

# ponytail: unbounded, one entry per distinct function content; add eviction if a long session edits steps thousands of times.
_DISPATCHERS: dict[str, Dispatcher] = {}
_REASONS: dict[tuple, str | None] = {}

# Numba keys a disk-cached step by its own bytecode only, so changes to a
# reachable helper would otherwise keep serving stale machine code.
SALT = hashlib.sha256(Path(cpython.__file__).read_bytes()).hexdigest()


class _SaltedCache(FunctionCache):
    def _index_key(self, sig, codegen):
        return (*super()._index_key(sig, codegen), SALT, fingerprint(self._py_func))


def numpy_dtype(annotation: Any) -> np.dtype:
    """The numpy dtype a value declared `annotation` is stored as between nodes.

    Follows `feature_kind`: `float`, `int`, `bool` and `str` (a dictionary
    code) are float64, int64, bool and int32; anything else, including
    `int | None`, is float64.

    >>> numpy_dtype(int), numpy_dtype(int | None)
    (dtype('int64'), dtype('float64'))
    """
    kind = feature_kind(annotation)
    # A `bytes` value is a span into Arrow memory, never stored between nodes.
    return KIND_DTYPES[FeatureKind.F64 if kind is FeatureKind.STR else kind]


def jit(fn: Callable) -> tuple[str, Dispatcher]:
    """The content key of `fn` and its njit dispatcher, shared by every function of the same content.

    Example::

        key, dispatcher = jit(disposable_income)
    """
    key = fingerprint(fn)
    if isinstance(fn, Dispatcher):
        return key, _DISPATCHERS.setdefault(key, fn)
    dispatcher = _DISPATCHERS.get(key)
    if dispatcher is None:
        # numba's disk cache needs a real source file, and never hits for a closure.
        dispatcher = _DISPATCHERS[key] = njit(fn)
        if fn.__closure__ is None and os.path.isfile(fn.__code__.co_filename):
            dispatcher._cache = _SaltedCache(fn)
    return key, dispatcher


def compile_call(node: CallNode) -> tuple[str, Callable, str | None]:
    """Compile one scalar or row node now: `(content key, callable, fallback reason)`.

    The callable is the njit dispatcher, or the plain Python function when
    numba can't compile it (then the reason says why). Only numba's
    compile-failure errors fall back; anything else raises.

    Example::

        key, fn, reason = compile_call(plan.calls[0].node)
    """
    prepared, helper_reason = _prepare_function(node.fn) if node.kind == "scalar" else (node.fn, None)
    key, dispatcher = jit(node.fn if helper_reason is not None else prepared)
    if helper_reason is not None:
        return key, dispatcher.py_func, helper_reason
    # numba would type a `date` or `list` input as the float64 it can't be converted to.
    odd = next((i for i in node.inputs if base_annotation(i.annotation) not in (float, int, bool, str, bytes, Any)), None)
    if odd is not None:
        return key, dispatcher.py_func, f"reads '{odd.name}' as {odd.annotation}, which no kernel takes"
    # numba types a scalar `bytes` argument as an array, so `==` against a literal compares
    # elementwise rather than as a whole value; there's no kernel signature for that.
    semantic_bytes = next((i for i in node.inputs
                           if base_annotation(i.annotation) is bytes and not is_raw(i.annotation)), None)
    if semantic_bytes is not None and node.kind == "scalar":
        return key, dispatcher.py_func, (f"reads '{semantic_bytes.name}' as bytes: numba can't type a scalar "
                                          "bytes value, only a byte array, so no kernel can compare it whole")
    # A `Raw[str]` output is already an int code by the time the function returns it,
    # so only a true `str` output needs the fallback: no fused array stores a string.
    string_output = next((o for o in node.outputs
                          if base_annotation(o.annotation) is str and not is_raw(o.annotation)), None)
    sig = _probe_signature(node)
    if sig is None:
        if string_output is not None:
            return key, dispatcher, f"writes '{string_output.name}' as str, which no fused kernel stores"
        return key, dispatcher, None
    if (key, sig) not in _REASONS:
        try:
            dispatcher.compile(sig)
            _REASONS[key, sig] = None
        # Compiling runs no step code, so a NotImplementedError here is numba's bytecode reader
        # meeting an opcode it lacks (Python 3.14's LOAD_COMMON_CONSTANT, from `any(... for ...)`).
        except (*FALLBACK_ERRORS, NotImplementedError) as e:
            _REASONS[key, sig] = f"{type(e).__name__}: {e}"
    reason = _REASONS[key, sig]
    # A compiled function returning `str` still can't join a shared array kernel, but calling
    # it once per row (not its raw Python body) boxes the result back automatically.
    if reason is None and string_output is not None:
        return key, dispatcher, f"writes '{string_output.name}' as str, which no fused kernel stores"
    return key, (dispatcher if reason is None else dispatcher.py_func), reason


def _prepare_function(fn: Callable, stack: tuple[int, ...] = ()) -> tuple[Callable, str | None]:
    """Replace declared helpers in `fn` globals with shared compiled dispatchers."""
    if isinstance(fn, Dispatcher):
        return fn, None
    if id(fn) in stack:
        return fn, f"recursive helper call involving '{fn.__name__}' cannot be compiled"
    globals_ = dict(fn.__globals__)
    changed = False
    for name in fn.__code__.co_names:
        called = globals_.get(name)
        if not isinstance(called, py_types.FunctionType):
            continue
        if is_python_only(called):
            return fn, f"calls python_only function '{called.__name__}', which runs in Python"
        signatures = helper_signatures(called)
        if signatures is None:
            return fn, f"calls '{called.__name__}' without @helper or @python_only"
        prepared, reason = _prepare_function(called, stack + (id(fn),))
        if reason is not None:
            return fn, f"helper '{called.__name__}': {reason}"
        key, dispatcher = _compile_helper(prepared, signatures)
        reason = _REASONS.get((key, "helper"))
        if reason is not None:
            return fn, f"helper '{called.__name__}' could not compile: {reason}"
        globals_[name] = dispatcher
        changed = True
    if not changed:
        return fn, None
    return py_types.FunctionType(fn.__code__, globals_, fn.__name__, fn.__defaults__, fn.__closure__), None


def _compile_helper(fn: Callable, signatures: tuple[tuple[tuple[type, ...], type], ...]) -> tuple[str, Dispatcher]:
    key, dispatcher = jit(fn)
    if (key, "helper") in _REASONS:
        return key, dispatcher
    try:
        for inputs, _ in signatures:
            dispatcher.compile(tuple(_numba_type(t) for t in inputs))
        _REASONS[key, "helper"] = None
    except (*FALLBACK_ERRORS, NotImplementedError) as e:
        _REASONS[key, "helper"] = f"{type(e).__name__}: {e}"
    return key, dispatcher


def _numba_type(annotation: type) -> Any:
    if is_raw(annotation):
        annotation = raw_base(annotation)
        if annotation is str:
            return types.int32
        if annotation is bytes:
            return SPAN
    if annotation is float:
        return types.float64
    if annotation is int:
        return types.int64
    if annotation is bool:
        return types.boolean
    if annotation is str:
        return types.unicode_type
    raise TypeError(f"unsupported @helper signature type {annotation!r}; use float, int or bool")


def parameters(fn: Callable) -> tuple[str, ...]:
    """The argument names of `fn` (or of a dispatcher's Python function), in order.

    >>> parameters(lambda income, cap: income)
    ('income', 'cap')
    """
    return tuple(inspect.signature(getattr(fn, "py_func", fn)).parameters)


def default_bundle(node: CallNode) -> tuple:
    """The params bundle of `node` with every param at its default; `()` for a node without params.

    Example::

        default_bundle(node).cap  # 48.0
    """
    return NodeParams(node.origin.path, node.params).defaults if node.params else ()


SPAN = types.UniTuple(types.int64, 2)


def _input_type(inp: Input) -> Any:
    raw = is_raw(inp.annotation)
    annotation = base_annotation(inp.annotation)
    if annotation is bytes and raw:
        return SPAN
    if annotation is bytes:
        # A null span has length -1, so a `bytes` input is never an Optional.
        return SPAN
    if annotation is str and raw:
        return types.int32
    if annotation is str:
        return types.Optional(types.unicode_type) if inp.null_policy is NullPolicy.OPTIONAL else types.unicode_type
    # An OPTIONAL `T | None` arrives as T's array plus a mask, so it is typed `Optional(T)`.
    t = from_dtype(numpy_dtype(base_annotation(inp.annotation)))
    return types.Optional(t) if inp.null_policy is NullPolicy.OPTIONAL else t


def _probe_signature(node: CallNode) -> tuple | None:
    # Typed as the values the kernel will pass; None when a value has no numba
    # type (a table param), and the node then compiles on first use instead.
    if any(d.schema is not None for d in node.params):
        return None
    try:
        bundle, consts = default_bundle(node), tuple(v for _, v in node.consts)
        ins = [_input_type(i) for i in node.inputs]
        if node.kind == "row":
            if SPAN in ins and node.params:
                # A `str` param of a node reading `bytes` reaches the kernel as a span of its UTF-8 bytes.
                bundle = bundle._replace(**{d.name: (0, 0) for d in node.params if d.annotation is str})
            return (types.Tuple(tuple(ins)), typeof(bundle), typeof(consts))
        by_arg = {i.arg: t for i, t in zip(node.inputs, ins)}
        # A `str` param reaches a scalar kernel as the int32 code of its literal.
        by_arg |= {d.arg: types.int32 if d.annotation is str else typeof(v) for d, v in zip(node.params, bundle)}
        by_arg |= {name: typeof(v) for name, v in node.consts}
        return tuple(by_arg[p] for p in parameters(node.fn))
    except (ValueError, KeyError):
        return None
