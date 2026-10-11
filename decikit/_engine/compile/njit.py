from __future__ import annotations

import datetime as dt
import dis
import inspect
import os
import types as py_types
from typing import Any, Callable, Literal, get_args, get_origin

import numpy as np
import typing_extensions as te
from numba import from_dtype, njit, typeof
from numba.core import types
from numba.core.caching import FunctionCache, IndexDataCacheFile, NullCache
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError, UnsupportedBytecodeError

# Importing `semantics` registers the overloads that make kernel arithmetic and span operations answer
# as CPython does; its SALT keys numba's disk cache.
from decikit._engine.compile.semantics import SALT
from decikit._engine.concurrency import Cache, once
from decikit._engine.compile.declared import tuple_mismatch
from decikit._engine.compile.fingerprint import fingerprint
from decikit._engine.compile.rows import rows_probe
from decikit._engine.compile.span import SPAN
from decikit._engine.compile.structs import bad_field, struct_dtype, struct_schema
from decikit._engine.ir.decls import (KIND_DTYPES, FeatureKind, Input, NullPolicy, Output, base_annotation,
                                     feature_kind, nullable)
from decikit._engine.ir.nodes import CallNode
from decikit._engine.params import NodeParams
from decikit.steps._helpers import allows_fallback, helper_signatures
from decikit.ops.nested import struct_item
from decikit._engine.compile.decimal_type import DecimalType
from decikit._engine.types import decimal_scale, is_raw, raw_base, columnar_item, item_schema

# UnsupportedBytecodeError (e.g. an `import` inside a step) isn't a NumbaError.
FALLBACK_ERRORS = (NumbaError, UnsupportedBytecodeError)

# Compiling runs no step code, so any of these is numba's own refusal: a NotImplementedError from its
# bytecode reader meeting an opcode it lacks, a bare assert from its lowering (a list of int and
# float), or a TypeError from boxing a literal it can't hash (dicts of mixed value types in a list).
# Runtime failures keep the narrower FALLBACK_ERRORS, which never swallows a step's own error.
_COMPILE_ERRORS = (*FALLBACK_ERRORS, NotImplementedError, AssertionError, TypeError)

_DECLARED_CALLEE = "calls the @allow_fallback function"

# ponytail: unbounded, one entry per distinct function content; add eviction if a long session edits steps thousands of times.
_DISPATCHERS = Cache()
_REASONS = Cache()


class _TolerantIndexFile(IndexDataCacheFile):
    def _load_index(self):
        # An index from another decikit version can name a type that has since moved; numba
        # unpickles the whole index before it compares keys, so treat any failure as empty.
        try:
            return super()._load_index()
        except Exception:
            return {}


class _SaltedCache(FunctionCache):
    def __init__(self, py_func):
        super().__init__(py_func)
        f = self._cache_file
        self._cache_file = _TolerantIndexFile(f._cache_path, self._impl.filename_base, f._source_stamp)

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
    dispatcher = once(_DISPATCHERS, key, lambda: _attach_cache(njit(fn), fn))
    # Dispatchers are shared by content, so the first sight of this code may have been a notebook
    # cell or an exec'd block: attach the cache as soon as any copy of it does have a file, or a
    # step stays uncacheable for the life of the process and `decikit build` warms nothing for it.
    if isinstance(dispatcher._cache, NullCache):
        with _DISPATCHERS.lock:
            _attach_cache(dispatcher, fn)
    return key, dispatcher


def _attach_cache(dispatcher: Dispatcher, fn: Callable) -> Dispatcher:
    # numba's disk cache needs a real source file and never hits for a closure. Attached before the
    # dispatcher is shared, so no thread compiles through it half set up.
    if isinstance(dispatcher._cache, NullCache) and fn.__closure__ is None and os.path.isfile(fn.__code__.co_filename):
        dispatcher._cache = _SaltedCache(fn)
    return dispatcher


def _refused(compile: Callable, *args: Any) -> str | None:
    # Why numba refused to compile, or None when it did.
    try:
        compile(*args)
    except _COMPILE_ERRORS as e:
        return _why(e)
    return None


def compile_call(node: CallNode) -> tuple[str, Callable, str | None, bool]:
    """Compile one scalar or row node now: `(content key, callable, fallback reason, declared)`.

    The callable is the njit dispatcher, or the plain Python function when
    numba can't compile it (then the reason says why). `declared` is true when
    `@allow_fallback` says the author accepts the fallback. Only numba's
    compile-failure errors fall back; anything else raises.

    Example::

        key, fn, reason, declared = compile_call(plan.calls[0].node)
    """
    prepared, helper_reason, undeclared = (_prepare_function(node.fn) if node.kind == "scalar"
                                           else (node.fn, None, ()))
    declared = allows_fallback(node.fn) or (helper_reason or "").startswith(_DECLARED_CALLEE)
    key, dispatcher = jit(node.fn if helper_reason is not None else prepared)
    if helper_reason is not None:
        return key, dispatcher.py_func, helper_reason, declared
    star = next((p for p in inspect.signature(getattr(node.fn, "py_func", node.fn)).parameters.values()
                 if node.kind == "scalar" and p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)), None)
    if star is not None:
        return key, dispatcher.py_func, (f"declares '{'**' if star.kind is star.VAR_KEYWORD else '*'}{star.name}', "
                                                    "which no kernel can bind columns to"), declared
    columnar_inputs = [i for i in node.inputs if columnar_item(i.annotation) is not None]
    struct_inputs = [i for i in node.inputs if struct_item(i.annotation) is not None]
    struct_reason = _struct_reason(node, struct_inputs)
    if struct_reason is not None:
        return key, dispatcher.py_func, struct_reason, declared
    # numba would type a `date` or `list` input as the float64 it can't be converted to.
    odd = next((i for i in node.inputs if i not in columnar_inputs and i not in struct_inputs
                and base_annotation(i.annotation) not in (float, int, bool, str, bytes, Any)
                and (node.kind == "row" or decimal_scale(i.annotation) is None)), None)
    if odd is not None:
        return key, dispatcher.py_func, f"reads '{odd.name}' as {odd.annotation}, which no kernel takes", declared
    # numba types a scalar `bytes` argument as an array, so `==` against a literal compares
    # elementwise rather than as a whole value; there's no kernel signature for that.
    semantic_bytes = next((i for i in node.inputs
                           if base_annotation(i.annotation) is bytes and not is_raw(i.annotation)), None)
    if semantic_bytes is not None and node.kind == "scalar":
        return key, dispatcher.py_func, (f"reads '{semantic_bytes.name}' as bytes: numba can't type a scalar "
                                         "bytes value, only a byte array, so no kernel can compare it whole"), declared
    # No array kernel can hold a variable-length string: numba's array of strings is fixed width, so its
    # element type would move with the longest value in the batch and recompile on every new one. Calling a
    # compiled dispatcher once per row instead costs more than the Python body it replaces, on a batch and
    # on one record, so a semantic `str` runs in Python, and so does a function step's `Literal`
    # output: only a tree or table returns one as the index of its value.
    no_kernel = (next((f"writes '{o.name}' as str, which no kernel stores" for o in node.outputs
                       if base_annotation(o.annotation) is str
                       or (node.kind == "scalar" and get_origin(base_annotation(o.annotation)) is Literal)), None)
                 or next((f"reads '{i.name}' as str, which no kernel holds" for i in node.inputs
                          if base_annotation(i.annotation) is str), None))
    if no_kernel is not None:
        return key, dispatcher.py_func, no_kernel, declared
    # A function step returning a dict, a TypedDict, a list, a tuple, a set or an array compiles alone, but no
    # kernel column stores it: inside a kernel it fails at launch and takes every other step of the kernel with it.
    unstored = next((_unstored(o.name, base_annotation(o.annotation)) for o in node.outputs
                     if node.kind == "scalar" and not _stored(base_annotation(o.annotation))), None)
    sig = _probe_signature(node)
    if sig is None:
        return key, dispatcher if unstored is None else dispatcher.py_func, unstored, declared
    reason = once(_REASONS, (key, sig), lambda: _refused(dispatcher.compile, sig))
    if reason is not None and undeclared:
        reason = f"calls '{undeclared[0]}' without @helper or @allow_fallback: {reason}"
    if unstored is not None:
        reason = unstored if reason is None else f"{unstored}; numba also refused the body: {reason}"
    # A nullable struct output has no record array with a per-row validity mask a kernel stores.
    nullable_struct = next((o for o in node.outputs
                            if nullable(o.annotation) and struct_item(base_annotation(o.annotation)) is not None), None)
    if reason is None and nullable_struct is not None:
        reason = (f"writes '{nullable_struct.name}' as Struct[...] | None, which no kernel stores; "
                  "return a plain Struct[...] instead")
    # A `Struct[Item]` output compiles when the step returns a tuple of its fields; a dict return
    # has no record a kernel stores, so the step stays on the Python path it has today.
    struct = struct_item(base_annotation(node.outputs[0].annotation)) if len(node.outputs) == 1 else None
    if reason is None and struct is not None:
        ret = dispatcher.overloads[sig].signature.return_type
        if isinstance(ret, types.DictType):
            reason = (f"writes '{node.outputs[0].name}' as Struct[...] by returning a dict, which no "
                      "kernel stores; return a tuple of the fields to compile it")
        elif isinstance(ret, types.BaseTuple):
            reason = tuple_mismatch(node.outputs[0], struct, ret.types)
    # Scalar only: the wrapper calls the dispatcher with the step's own positional args, which is
    # not how a row node's `fn(row, params, consts)` is called.
    ragged_item = (columnar_item(base_annotation(node.outputs[0].annotation))
                   if node.kind == "scalar" and len(node.outputs) == 1 else None)
    if reason is None and ragged_item is not None:
        reason = _ragged_reason(node.outputs[0], ragged_item, dispatcher, sig)
    # A `Columnar[...]` input joins the shared array kernel as flat per-field arrays sliced per row,
    # unless it's OPTIONAL: one kernel has one signature, and no kernel value is both a namedtuple
    # of views and `None`. A row node (a tree/table/scorecard feature) keeps the per-row dispatcher
    # too: nothing wires a ragged source into a row node's feature tuple yet.
    if reason is None and columnar_inputs and not (
            node.kind == "scalar" and all(i.null_policy is not NullPolicy.OPTIONAL for i in columnar_inputs)):
        names = ", ".join(f"'{i.name}'" for i in columnar_inputs)
        return key, dispatcher, (f"reads {names} as Columnar[...], which runs one call per row, "
                                 "outside the shared kernel"), declared
    # A `Columnar[Item]` output has no fixed-stride kernel column of its own (a row emits a variable
    # number of items); the wrapper drains the step's returned list into a shared growable buffer and
    # reports back how many items that row added, an ordinary int64 scalar a kernel already stores.
    if reason is None and ragged_item is not None:
        return (*jit(_ragged_wrapper(dispatcher)), None, declared)
    return key, (dispatcher if reason is None else dispatcher.py_func), reason, declared


def _stored(annotation: Any) -> bool:
    # What no kernel column holds: a container (an array would be cut to its first element), a time or bytes
    # (stored as the float64 of a body that only returns None). A column does hold `Any`, a numpy number or a
    # NewType of float, which a list of what it stores would refuse.
    container = (get_origin(annotation) or annotation) in (dict, list, tuple, set, frozenset, np.ndarray)
    return not (container or te.is_typeddict(annotation) or annotation in _UNSTORED)


_UNSTORED = (bytes, dt.date, dt.datetime, dt.time, dt.timedelta, np.datetime64, np.timedelta64)


def _unstored(name: str, annotation: Any) -> str:
    # Names the declared form of the same value that does compile, where one exists.
    origin = get_origin(annotation) or annotation
    args = get_args(annotation)
    if origin is dict or te.is_typeddict(annotation):
        typed = te.is_typeddict(annotation)
        item = annotation.__name__ if typed else "Item"
        fix = (f"to compile it, declare -> Struct[{item}] and return a tuple of its fields (one record), "
               f"-> Columnar[{item}] and return a list of such tuples (many rows), or return a tuple under "
               "@step(outputs=(...)) (one output per value)")
        fix += "" if typed else ", Item a TypedDict of the dict's keys"
    elif origin is tuple:
        fix = ("to compile it, return the tuple under @step(outputs=(...)) (one output per element), "
               "or declare -> Struct[Item] (one record)")
    elif origin in (list, np.ndarray):
        item = args[0].__name__ if args and te.is_typeddict(args[0]) else "Item"
        fix = (f"to compile it, declare -> Columnar[{item}] and return a list of tuples of its fields "
               f"(a one-field {item} for a list of numbers)")
    else:
        fix = "no compiled step returns one; accept it with @allow_fallback"
    return f"writes '{name}' as {_shown(annotation)}, which no kernel stores; {fix}"


def _shown(annotation: Any) -> str:
    args = get_args(annotation)
    if not args:
        return getattr(annotation, "__name__", str(annotation))
    return f"{_shown(get_origin(annotation))}[{', '.join(_shown(a) for a in args)}]"


def _ragged_reason(output: Output, item: Any, dispatcher: Dispatcher, sig: tuple) -> str | None:
    if nullable(output.annotation):
        return (f"writes '{output.name}' as Columnar[...] | None, which no kernel stores; "
                "return a plain Columnar[...] instead")
    schema = struct_schema(item)
    bad = bad_field(schema)
    if bad is not None:
        return (f"writes '{output.name}' item field '{bad[0]}' as {bad[1]}, which no kernel stores; "
                "a Columnar[...] output item must be float, int or bool")
    ret = dispatcher.overloads[sig].signature.return_type
    fields = ret.dtype.types if isinstance(ret, types.List) and isinstance(ret.dtype, types.BaseTuple) else None
    if fields is None:
        return (f"writes '{output.name}' as Columnar[...] by returning a list of dicts, which no "
                "kernel stores; return a list of tuples to compile it")
    return tuple_mismatch(output, item, fields)


def _ragged_wrapper(dispatcher: Dispatcher) -> Callable:
    # Fixed arity (`packed`, `sink`), not `*args`: `kernel.py`'s hand-rolled IR calls this with one
    # value per argument it was typed with, which a `*args` signature resolves against a single
    # packed tuple type instead -- a mismatch a normal njit call site never hits. `packed` is the
    # step's own arguments, already packed into one tuple by the caller (mirroring a row node's
    # `SrcKind.ROW`); a plain function, not `@njit` here, so `jit()` compiles it once its own
    # content (which closes over `dispatcher`) is fingerprinted, exactly like any other step.
    def wrapper(packed, sink):
        for item in dispatcher(*packed):
            sink.push(item)
        return sink.length

    return wrapper


def _struct_reason(node: CallNode, inputs: list[Input]) -> str | None:
    # A record has no per-field validity and no variable-length field, so a fill, an
    # Optional or a field of another type keeps the step on the Python path it has today.
    for inp in inputs:
        schema = struct_schema(struct_item(inp.annotation))
        bad = bad_field(schema)
        if bad is not None:
            return (f"reads field '{inp.name}.{bad[0]}' as {bad[1]}, which no kernel record holds; "
                    "a struct field must be float, int or bool")
        if inp.null_policy is not NullPolicy.REQUIRED:
            return (f"reads '{inp.name}' as Struct[...] with {inp.null_policy.value} nulls, which a kernel "
                    "record can't hold: it has no per-field validity")
    return None


def _prepare_function(fn: Callable, stack: tuple[int, ...] = ()) -> tuple[Callable, str | None, tuple[str, ...]]:
    """`fn` with declared helpers pointed at shared dispatchers, why it can't compile, undeclared callees."""
    if isinstance(fn, Dispatcher):
        return fn, None, ()
    if id(fn) in stack:
        return fn, f"recursive helper call involving '{fn.__name__}' cannot be compiled", ()
    globals_ = dict(fn.__globals__)
    changed = False
    undeclared: list[str] = []
    for name, attr in sorted(_global_refs(fn.__code__), key=repr):
        owner = globals_.get(name)
        in_module = isinstance(owner, py_types.ModuleType)
        called = owner.__dict__.get(attr) if in_module else owner
        if not isinstance(called, py_types.FunctionType):
            continue
        if allows_fallback(called):
            return fn, f"{_DECLARED_CALLEE} '{called.__name__}', which runs in Python", ()
        signatures = helper_signatures(called)
        if signatures is None:
            # numba compiles what it can (an @overload, a supported numpy function) and only
            # needs this name when it fails.
            undeclared.append(called.__name__)
            continue
        prepared, reason, _ = _prepare_function(called, stack + (id(fn),))
        if reason is not None:
            return fn, f"helper '{called.__name__}': {reason}", ()
        key, dispatcher = _compile_helper(prepared, signatures)
        reason = _REASONS.get((key, "helper"))
        if reason is not None:
            return fn, f"helper '{called.__name__}' could not compile: {reason}", ()
        if not in_module:
            globals_[name] = dispatcher
        else:
            # A copy, so the module itself keeps its plain function.
            if owner is fn.__globals__[name]:
                owner = globals_[name] = py_types.ModuleType(owner.__name__)
                owner.__dict__.update(fn.__globals__[name].__dict__)
            setattr(owner, attr, dispatcher)
        changed = True
    names = tuple(sorted(undeclared))
    if not changed:
        return fn, None, names
    return py_types.FunctionType(fn.__code__, globals_, fn.__name__, fn.__defaults__, fn.__closure__), None, names


def _global_refs(code: py_types.CodeType) -> set[tuple[str, str | None]]:
    # LOAD_GLOBAL only: `co_names` also holds attribute names, and `rates.rate` is not a call to `rate`.
    # A global followed by an attribute is `(module, attribute)`: `caps.round_down(x)` reads a function off a module.
    refs: set[tuple[str, str | None]] = set()
    ins = list(dis.get_instructions(code))
    for k, i in enumerate(ins):
        if i.opname == "LOAD_GLOBAL":
            nxt = ins[k + 1] if k + 1 < len(ins) else None
            refs.add((i.argval, nxt.argval if nxt is not None and nxt.opname in ("LOAD_ATTR", "LOAD_METHOD") else None))
    for const in code.co_consts:
        if isinstance(const, py_types.CodeType):
            refs |= _global_refs(const)
    return refs


def _compile_helper(fn: Callable, signatures: tuple[tuple[tuple[type, ...], type], ...]) -> tuple[str, Dispatcher]:
    key, dispatcher = jit(fn)

    def compile_all() -> None:
        for inputs, _ in signatures:
            dispatcher.compile(tuple(_numba_type(t) for t in inputs))

    once(_REASONS, (key, "helper"), lambda: _refused(compile_all))
    return key, dispatcher


def _why(e: BaseException) -> str:
    # numba's lowering asserts carry no message at all.
    return f"{type(e).__name__}: {str(e) or 'numba could not compile this step'}"


def _numba_type(annotation: type) -> Any:
    if (scale := decimal_scale(annotation)) is not None:
        return DecimalType(scale)
    if is_raw(annotation):
        annotation = raw_base(annotation)
        if annotation is bytes:
            return SPAN
    if annotation is float:
        return types.float64
    if annotation is int:
        return types.int64
    if annotation is bool:
        return types.boolean
    raise TypeError(f"unsupported @helper signature type {annotation!r}; use float, int, bool or a Decimal")


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
    item = columnar_item(inp.annotation)
    if item is not None:
        return typeof(rows_probe(item_schema(item)))
    item = struct_item(inp.annotation)
    if item is not None:
        return from_dtype(struct_dtype(struct_schema(item)))
    annotation = base_annotation(inp.annotation)
    if (scale := decimal_scale(inp.annotation)) is not None:
        return types.Optional(DecimalType(scale)) if inp.null_policy is NullPolicy.OPTIONAL else DecimalType(scale)
    if annotation is bytes:
        # A null span has length -1, so a `bytes` input is never an Optional.
        return SPAN
    # An OPTIONAL `T | None` arrives as T's array plus a mask, so it is typed `Optional(T)`.
    t = from_dtype(numpy_dtype(base_annotation(inp.annotation)))
    return types.Optional(t) if inp.null_policy is NullPolicy.OPTIONAL else t


def _param_type(decl: Any, value: Any) -> Any:
    scale = decimal_scale(decl.annotation)
    return typeof(value) if scale is None else DecimalType(scale)


def _has_span(ins: list) -> bool:
    # A top-level `bytes` input is typed `SPAN` directly; a `Columnar[...]` input is a namedtuple
    # whose `str` fields are each an array of `SPAN`, so `SPAN in ins` alone misses it.
    return any(t is SPAN or (isinstance(t, types.BaseNamedTuple)
                             and any(getattr(f, "dtype", None) is SPAN for f in t.types))
              for t in ins)


def _probe_signature(node: CallNode) -> tuple | None:
    # Typed as the values the kernel will pass; None when a value has no numba
    # type (a table param), and the node then compiles on first use instead.
    if any(d.schema is not None for d in node.params):
        return None
    try:
        bundle, consts = default_bundle(node), tuple(v for _, v in node.consts)
        ins = [_input_type(i) for i in node.inputs]
        if node.kind == "row":
            if _has_span(ins) and node.params:
                # A `str` param of a node reading `bytes` reaches the kernel as a span of its UTF-8 bytes.
                bundle = bundle._replace(**{d.name: (0, 0) for d in node.params if d.annotation is str})
            return (types.Tuple(tuple(ins)), typeof(bundle), typeof(consts))
        by_arg = {i.arg: t for i, t in zip(node.inputs, ins)}
        # A `str` param of a node reading `bytes` (or a `Columnar[...]` item field) reaches the kernel
        # as a span of its UTF-8 bytes; otherwise as the int32 code of its literal.
        par = types.UniTuple(types.int64, 2) if _has_span(ins) else types.int32
        by_arg |= {d.arg: par if d.annotation is str else _param_type(d, v) for d, v in zip(node.params, bundle)}
        by_arg |= {name: typeof(v) for name, v in node.consts}
        return tuple(by_arg[p] for p in parameters(node.fn))
    except (ValueError, KeyError):
        return None
