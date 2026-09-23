from __future__ import annotations

import inspect
import os
from typing import Any, Callable

import numpy as np
from numba import from_dtype, njit, typeof
from numba.core import types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError, UnsupportedBytecodeError

from decider.engine.compile.fingerprint import fingerprint
from decider.engine.ir.decls import FeatureKind, Input, NullPolicy, feature_kind
from decider.engine.ir.nodes import CallNode
from decider.engine.params import NodeParams

# UnsupportedBytecodeError (e.g. an `import` inside a step) isn't a NumbaError.
FALLBACK_ERRORS = (NumbaError, UnsupportedBytecodeError)

_DTYPES = {FeatureKind.F64: np.dtype(np.float64), FeatureKind.I64: np.dtype(np.int64),
           FeatureKind.BOOL: np.dtype(np.bool_), FeatureKind.CODE: np.dtype(np.int32)}

# ponytail: unbounded, one entry per distinct function content; add eviction if a long session edits steps thousands of times.
_DISPATCHERS: dict[str, Dispatcher] = {}
_REASONS: dict[tuple, str | None] = {}


def numpy_dtype(annotation: Any) -> np.dtype:
    """The numpy dtype a value declared `annotation` is stored as between nodes.

    Follows `feature_kind`: `float`, `int`, `bool` and `str` (a dictionary
    code) are float64, int64, bool and int32; anything else, including
    `int | None`, is float64.

    >>> numpy_dtype(int), numpy_dtype(int | None)
    (dtype('int64'), dtype('float64'))
    """
    return _DTYPES.get(feature_kind(annotation), _DTYPES[FeatureKind.F64])


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
        cache = fn.__closure__ is None and os.path.isfile(fn.__code__.co_filename)
        dispatcher = _DISPATCHERS[key] = njit(cache=cache)(fn)
    return key, dispatcher


def compile_call(node: CallNode) -> tuple[str, Callable, str | None]:
    """Compile one scalar or row node now: `(content key, callable, fallback reason)`.

    The callable is the njit dispatcher, or the plain Python function when
    numba can't compile it (then the reason says why). Only numba's
    compile-failure errors fall back; anything else raises.

    Example::

        key, fn, reason = compile_call(plan.calls[0].node)
    """
    key, dispatcher = jit(node.fn)
    sig = _probe_signature(node)
    if sig is None:
        return key, dispatcher, None
    if (key, sig) not in _REASONS:
        try:
            dispatcher.compile(sig)
            _REASONS[key, sig] = None
        except FALLBACK_ERRORS as e:
            _REASONS[key, sig] = f"{type(e).__name__}: {e}"
    reason = _REASONS[key, sig]
    return key, (dispatcher if reason is None else dispatcher.py_func), reason


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


def _input_type(inp: Input) -> Any:
    t = from_dtype(numpy_dtype(inp.annotation))
    return types.Optional(t) if inp.null_policy is NullPolicy.OPTIONAL else t


def _probe_signature(node: CallNode) -> tuple | None:
    # Typed as the values the kernel will pass; None when a value has no numba
    # type (a table param), and the node then compiles on first use instead.
    try:
        bundle, consts = default_bundle(node), tuple(v for _, v in node.consts)
        ins = [_input_type(i) for i in node.inputs]
        if node.kind == "row":
            return (types.Tuple(tuple(ins)), typeof(bundle), typeof(consts))
        by_arg = {i.arg: t for i, t in zip(node.inputs, ins)}
        by_arg |= {d.name: typeof(v) for d, v in zip(node.params, bundle)}
        by_arg |= {name: typeof(v) for name, v in node.consts}
        return tuple(by_arg[p] for p in parameters(node.fn))
    except (ValueError, KeyError):
        return None
