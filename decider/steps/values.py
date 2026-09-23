from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict

from decider.serializable.dataframe import DataFrame

T = t.TypeVar("T")


class ParamRef(BaseModel):
    """A config value read from the params document instead of written inline.

    `{"param": "hi_thresh", "default": 0.7}` is a local param of the step;
    `{"param": "base_rate", "shared": true}` reads `shared.base_rate`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    default: t.Any = None
    shared: bool = False


class TableRef(BaseModel):
    """A table-valued config field whose rows come from the params document.

    `{"table": "prices"}` reads the rows of param `prices`; the owning step
    declares the columns and dtypes, so editing rows never recompiles.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    shared: bool = False


Value = t.Union[T, ParamRef]
"""`Value[float]` accepts a literal (`0.7`) or a `ParamRef` dict."""

TableValue = t.Union[DataFrame, TableRef]
"""Inline rows (`{"data": [...]}`) or a `TableRef`."""
