from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict

from decider.serializable.dataframe import DataFrame

T = t.TypeVar("T")


class ParamRef(BaseModel):
    """A config value read from the params document instead of written inline.

    `{"param": "hi_thresh", "default": 0.7}` is a local param of the step;
    `{"param": "base_rate", "shared": true}` reads `shared.base_rate`.

    Example::

        ThresholdRule(name="hi", column="score", threshold=ParamRef(param="hi_thresh", default=0.7))
        # params document: {"hi": {"hi_thresh": 0.8}}
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    default: t.Any = None
    shared: bool = False


class TableRef(BaseModel):
    """A table-valued config field whose rows come from the params document.

    `{"table": "prices"}` makes a required param `prices` of the owning step,
    which declares the columns and dtypes, so editing rows or their count
    never recompiles. In the params document the rows sit under the step's
    node path, as a list with one dict per row and a value for every column.
    `{"table": "prices", "shared": true}` reads `shared.prices` instead, so
    several steps can use one table. `parameters()` reports the param as
    `{"type": "table", "schema": {column: dtype}}`; `defaults()` leaves it
    out, since it has no default.

    Example::

        DecisionTableConfig(name="pricing", rows=TableRef(table="prices"), ...)
        # inside flow(..., name="loans"), the params document is
        # {"loans": {"pricing": {"prices": [{"product": "loan", "rate": 0.1}, {"product": "card", "rate": 0.2}]}}}
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    shared: bool = False


Value = t.Union[T, ParamRef]
"""`Value[float]` accepts a literal (`0.7`) or a `ParamRef` dict."""

TableValue = t.Union[DataFrame, TableRef]
"""A table field: rows inline (a list of row dicts, or `{"data": [...]}`), or `{"table": "<param>"}` for rows
from the params document (see `TableRef`)."""
