from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import Field

from decider.engine.ir.decls import OnInvalid

# A default of one of these types is returned as a subclass of its own type, so
# a plain function using it as a default still works when called directly.
_CARRIER_TYPES = (float, int, str, list, dict, tuple)
_NO_DEFAULT = object()


class ParamSpec:
    """What `param()` returns: the default itself (when its type allows) plus the declaration."""

    default: Any
    field_info: Any
    required: bool
    shared_key: str | None
    on_invalid: OnInvalid
    # A table's `(column, type name)` pairs; `None` for a plain param.
    schema: tuple[tuple[str, str], ...] | None = None

    # A bool/None default can't be subclassed, so a direct call sees this marker: make it test like its value.
    def __bool__(self) -> bool:
        return bool(self.default)


class MissingAs:
    """What `missing_as()` returns: the fill value itself (when its type allows) plus the marker."""

    fill: Any

    def __bool__(self) -> bool:
        return bool(self.fill)


@cache
def _carrier_class(kind: type, value_type: type) -> type:
    return type(f"{value_type.__name__}_{kind.__name__}", (value_type, kind), {})


def _mark(kind: type, value: Any, **attrs: Any) -> Any:
    obj = _carrier_class(kind, type(value))(value) if type(value) in _CARRIER_TYPES else kind()
    obj.__dict__.update(attrs)
    return obj


def is_plain_marker(value: Any) -> bool:
    """True for a `param()`/`missing_as()` result that is not also the value itself."""
    return type(value) in (ParamSpec, MissingAs)


def param(
    default: Any = _NO_DEFAULT,
    *,
    required: bool = False,
    shared_key: str | None = None,
    on_invalid: OnInvalid = "error",
    **field_kwargs: Any,
) -> Any:
    """Declare a tunable param as a function argument's default.

    Params are configuration from the params document, the same for every
    record of a run; they are not request data. A request field is a plain
    argument, read from the record by its name::

        def affordable(ratio: float,                           # request field / earlier step's output
                       min_ratio: float = param(0.3)) -> bool:  # config from the params document
            return ratio >= min_ratio

    Every extra keyword (`ge`, `le`, `description`, ...) is passed to pydantic's
    `Field`, so the params document is validated against it.

    Args:
        default: the value used when the params document doesn't set it. Any
            value works, including `None` and `bool`.
        required: declare a param with no default; a params document that
            doesn't set it is invalid.
        shared_key: read the value from the document's top-level `"shared"`
            entry under this key instead of from the step's own entry.
        on_invalid: `"error"` fails, `"warn"` records a warning and uses the
            default, `"default"` uses the default silently.

    Example::

        def cap_by_income(term_cap: float,
                          cap: float = param(48.0, ge=6, le=60),
                          base_rate: float = param(5.0, shared_key="base_rate")) -> float:
            return min(term_cap, cap)

        cap_by_income(60.0)  # 48.0: the function is still plain Python

    A `bool` or `None` default comes back as a marker that tests like its
    value (`if flag:` works on a direct call), but `flag is False` doesn't
    hold; write `if not flag:`. Param names must be Python identifiers.
    """
    if required == (default is not _NO_DEFAULT):
        raise TypeError("param() takes either a default or required=True, not both or neither")
    if on_invalid not in ("error", "warn", "default"):
        raise ValueError(f"on_invalid must be 'error', 'warn' or 'default', not {on_invalid!r}")
    info = Field(**field_kwargs) if required else Field(default, **field_kwargs)
    value = None if required else default
    return _mark(ParamSpec, value, default=value, field_info=info, required=required,
                 shared_key=shared_key, on_invalid=on_invalid)


def missing_as(value: Any) -> Any:
    """Declare that a null in this input is replaced by `value`.

    To let the null through instead, annotate the input `T | None`
    (`missing_as(None)` is refused). A `bool` fill comes back as a marker
    that tests like its value on a direct call; use `if not x:`, not `x is False`.

    Example::

        def score(bureau_score: float = missing_as(0.0),
                  deceased: bool = missing_as(False),
                  bonus: float | None = None) -> float:
            return bureau_score * 0.01
    """
    if value is None:
        raise TypeError("missing_as(None) fills a null with a null; annotate the input `T | None` instead")
    return _mark(MissingAs, value, fill=value)
