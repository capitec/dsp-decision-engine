from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import types
import typing
from typing import Any, Literal

OnInvalid = Literal["error", "warn", "default"]


class NullPolicy(Enum):
    """What a step's input does when the data holds a null.

    - `REQUIRED` (`x: float`): a null is an error.
    - `MISSING_AS` (`x: float = missing_as(0.0)`): a null is replaced by the fill.
    - `OPTIONAL` (`x: float | None`): the step receives `None`.
    """

    REQUIRED = "required"
    MISSING_AS = "missing_as"
    OPTIONAL = "optional"


class FeatureKind(IntEnum):
    """The typed row array a value of a given annotation is gathered into.

    The integer values are data: compiled trees store them per node and switch
    on them, so they never change.
    """

    F64 = 0
    I64 = 1
    BOOL = 2
    CODE = 3  # str: an int32 dictionary code
    STR = 4  # bytes: a raw string span (address, byte length; -1 for null)


_KIND_BY_ANNOTATION = {float: FeatureKind.F64, int: FeatureKind.I64, bool: FeatureKind.BOOL,
                       str: FeatureKind.CODE, bytes: FeatureKind.STR}


def feature_kind(annotation: Any) -> FeatureKind:
    """The `FeatureKind` for an annotation; anything unrecognised is `F64`.

    >>> feature_kind(int)
    <FeatureKind.I64: 1>
    """
    return _KIND_BY_ANNOTATION.get(annotation, FeatureKind.F64)


def base_annotation(annotation: Any) -> Any:
    """`T` for an optional `T | None`, else the annotation itself.

    >>> base_annotation(int | None)
    <class 'int'>
    """
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


@dataclass(frozen=True, slots=True)
class Input:
    """One declared input of a step. `fill` is set only for `MISSING_AS`.

    `name` is the column it reads, which a relabel may change; `arg` is the
    function argument it feeds, which never changes (default: `name`).

    Example::

        Input("monthly_net_salary", float, arg="net_income")
    """

    name: str
    annotation: Any
    null_policy: NullPolicy = NullPolicy.REQUIRED
    fill: Any = None
    arg: str | None = None

    def __post_init__(self) -> None:
        if self.arg is None:
            object.__setattr__(self, "arg", self.name)


@dataclass(frozen=True, slots=True)
class Output:
    """One declared output of a step."""

    name: str
    annotation: Any


@dataclass(frozen=True, slots=True)
class ParamDecl:
    """One tunable param of a node.

    Args:
        default: the plain default value; ignored when `required`.
        field_info: the pydantic `FieldInfo` carrying bounds and descriptions.
            `None` means no constraints.
        shared_key: the key under the document's top-level `"shared"` entry, or
            `None` for a param local to its node.
        on_invalid: what an invalid value does: `"error"` fails, `"warn"` and
            `"default"` fall back to the default (with and without a warning).
        schema: for a table-valued param, its `(column, dtype)` pairs
            (`(("product", "str"), ("rate", "float"))`); `None` otherwise.
        arg: the function argument a scalar call passes it as (default:
            `name`), for a config param whose document key differs from it.

    Example::

        ParamDecl("base_rate", float, 5.0, shared_key="base_rate", on_invalid="warn")
        ParamDecl("hi_thresh", float, 0.7, arg="threshold")   # {"hi_thresh": ...} feeds `threshold`
    """

    name: str
    annotation: Any
    # Defaults and FieldInfo may be unhashable (lists, dicts), so they stay out of hash and ==.
    default: Any = field(default=None, compare=False, hash=False)
    field_info: Any = field(default=None, compare=False, hash=False)
    required: bool = False
    shared_key: str | None = None
    on_invalid: OnInvalid = "error"
    schema: tuple[tuple[str, str], ...] | None = None
    arg: str | None = None

    def __post_init__(self) -> None:
        if self.arg is None:
            object.__setattr__(self, "arg", self.name)
