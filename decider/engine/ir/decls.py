from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
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


@dataclass(frozen=True, slots=True)
class Input:
    """One declared input of a step. `fill` is set only for `MISSING_AS`."""

    name: str
    annotation: Any
    null_policy: NullPolicy = NullPolicy.REQUIRED
    fill: Any = None


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

    Example::

        ParamDecl("base_rate", float, 5.0, shared_key="base_rate", on_invalid="warn")
    """

    name: str
    annotation: Any
    default: Any = None
    field_info: Any = None
    required: bool = False
    shared_key: str | None = None
    on_invalid: OnInvalid = "error"
