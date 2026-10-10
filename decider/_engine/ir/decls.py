from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import types
import typing
from typing import Any, Literal

import numpy as np
from decider._engine.types import annotation_cache, decimal_scale, is_decimal, is_raw, plain_annotation, raw_base, text_as_str, to_raw_input

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


# The array dtype of each kind; a STR value is two of them, (address, byte length).
KIND_DTYPES: dict[FeatureKind, np.dtype] = {
    FeatureKind.F64: np.dtype(np.float64), FeatureKind.I64: np.dtype(np.int64),
    FeatureKind.BOOL: np.dtype(np.bool_), FeatureKind.CODE: np.dtype(np.int32),
    FeatureKind.STR: np.dtype(np.int64),
}

# Annotations whose values live in typed arrays; anything else is a Python object.
TYPED = (float, int, bool)


def is_typed(annotation: Any) -> bool:
    """Whether values of `annotation` live in a typed array: float, int, bool, or a Decimal (an int64).

    >>> is_typed(float), is_typed(str)
    (True, False)
    """
    return annotation in TYPED or is_decimal(annotation)

_KIND_BY_ANNOTATION = {float: FeatureKind.F64, int: FeatureKind.I64, bool: FeatureKind.BOOL,
                       str: FeatureKind.CODE, bytes: FeatureKind.STR}


def feature_kind(annotation: Any) -> FeatureKind:
    """The `FeatureKind` for an annotation; anything unrecognised is `F64`.

    >>> feature_kind(int)
    <FeatureKind.I64: 1>
    """
    kind_annotation = raw_base(annotation) if is_raw(annotation) else plain_annotation(annotation)
    kind = _KIND_BY_ANNOTATION.get(kind_annotation)
    if kind is None:
        # A Decimal is its int64 mantissa, `Decimal | None` included.
        kind = FeatureKind.I64 if is_decimal(kind_annotation) else FeatureKind.F64
    return kind


@annotation_cache
def base_annotation(annotation: Any) -> Any:
    """`T` for an optional `T | None`, else the annotation itself.

    >>> base_annotation(int | None)
    <class 'int'>
    """
    if is_raw(annotation):
        return raw_base(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return plain_annotation(args[0])
    return annotation


def nullable(annotation: Any) -> bool:
    """Whether a value declared `annotation` may be `None` (`T | None`).

    >>> nullable(float | None), nullable(float)
    (True, False)
    """
    annotation = plain_annotation(annotation)
    return typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation)


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


def stored_fill(decl: Input) -> Any:
    """`decl.fill` as it is stored: the int64 of a Decimal fill, else the fill itself.

    >>> stored_fill(Input("x", int, NullPolicy.MISSING_AS, 7))
    7
    """
    scale = decimal_scale(decl.annotation)
    return decl.fill if scale is None or decl.fill is None else to_raw_input(decl.fill, scale)


@dataclass(frozen=True, slots=True)
class Output:
    """One declared output of a step."""

    name: str
    annotation: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotation", text_as_str(self.annotation))


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
        seeded_from: for a table, the config document its default rows were read from.
        allow_override: a params document may set this table although `seeded_from` names a document.

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
    seeded_from: str | None = None
    allow_override: bool = False

    def __post_init__(self) -> None:
        if self.arg is None:
            object.__setattr__(self, "arg", self.name)
        object.__setattr__(self, "annotation", text_as_str(self.annotation))
