from __future__ import annotations

import inspect
import typing
from typing import Any, Callable

from decider.engine.ir.decls import Input, NullPolicy, Output, ParamDecl, nullable
from decider.engine.params.declare import MissingAs, ParamSpec, is_plain_marker
from decider.exceptions import IRError

_EMPTY = inspect.Parameter.empty


def _outputs(fn: Callable, names: tuple[str, ...], ret: Any) -> tuple[Output, ...]:
    if len(names) == 1:
        return (Output(names[0], ret),)
    args = typing.get_args(ret) if typing.get_origin(ret) is tuple else ()
    if len(args) != len(names) or Ellipsis in args:
        raise IRError(
            f"{fn.__qualname__}: outputs {names} need a return annotation tuple[...] "
            f"of {len(names)} types, got {ret!r}"
        )
    return tuple(Output(n, a) for n, a in zip(names, args))


def harvest(
    fn: Callable, outputs: tuple[str, ...] | None = None
) -> tuple[tuple[Input, ...], tuple[ParamDecl, ...], tuple[Output, ...]]:
    """Read a function's inputs, params and outputs from its signature.

    - `x: float` is a required input; `x: float | None` an optional one;
      `x: float = missing_as(0.0)` fills nulls with `0.0`.
    - `x: float = param(...)` is a param.
    - `outputs` defaults to `(fn.__name__,)`. Several outputs need a
      `tuple[...]` return annotation of the same length.

    Example::

        inputs, params, outs = harvest(cap_by_income)
    """
    sig = inspect.signature(fn, eval_str=True)
    inputs: list[Input] = []
    params: list[ParamDecl] = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise IRError(f"{fn.__qualname__}: *args/**kwargs can't be step inputs; inputs are wired by name")
        ann = Any if p.annotation is _EMPTY else p.annotation
        d = p.default
        if isinstance(d, ParamSpec):
            if ann is Any and not d.required:
                ann = type(d.default)
            params.append(ParamDecl(name, ann, d.default, d.field_info, d.required, d.shared_key, d.on_invalid))
        elif isinstance(d, MissingAs):
            inputs.append(Input(name, type(d.fill) if ann is Any else ann, NullPolicy.MISSING_AS, d.fill, arg=name))
        elif nullable(ann):
            inputs.append(Input(name, ann, NullPolicy.OPTIONAL, arg=name))
        elif d is not _EMPTY:
            # A bare default could mean a tunable knob or a fill for nulls; make the author say which.
            raise IRError(
                f"{fn.__qualname__}: parameter '{name}' has a bare default ({d!r}). "
                f"Use param({d!r}) for a tunable param or missing_as({d!r}) to fill nulls."
            )
        else:
            inputs.append(Input(name, ann, NullPolicy.REQUIRED, arg=name))
    ret = Any if sig.return_annotation is _EMPTY else sig.return_annotation
    return tuple(inputs), tuple(params), _outputs(fn, outputs or (fn.__name__,), ret)


def call_with_defaults(fn: Callable, /, *args: Any, **kwargs: Any) -> Any:
    """Call `fn`, replacing any `param()`/`missing_as()` marker default it would receive with its value.

    Defaults of most types are already the value itself; `bool`, `None` and
    other types can't be, and `param(required=True)` has no value at all.

    Example::

        def gate(x: float, on: bool = param(True)) -> float:
            return x if on else 0.0

        call_with_defaults(gate, 2.0)  # 2.0
    """
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    for name, p in sig.parameters.items():
        if name in bound.arguments or not is_plain_marker(p.default):
            continue
        if isinstance(p.default, ParamSpec) and p.default.required:
            raise TypeError(f"{fn.__qualname__}() missing required param '{name}'")
        bound.arguments[name] = p.default.default if isinstance(p.default, ParamSpec) else p.default.fill
    return fn(*bound.args, **bound.kwargs)
