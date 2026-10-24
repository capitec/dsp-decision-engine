"""How a kernel stores what a step writes."""
from __future__ import annotations

import copy
import typing
from typing import Any

import numpy as np
import polars as pl

from decikit.ops.nested import struct_item
from decikit._engine.compile.njit import numpy_dtype
from decikit._engine.compile.structs import struct_dtype, struct_schema
from decikit._engine.ir.decls import base_annotation
from decikit._engine.types import columnar_item


def output_dtype(annotation: Any) -> np.dtype:
    """The dtype a kernel stores an output declared `annotation` in: `numpy_dtype` of `T` for `T | None`.

    A `Literal` of strings or a `lookup(...)` is stored as the int64 index of
    its value; a `Struct[Item]` as one record per row. A `Columnar[Item]` has
    no fixed-stride column of its own: this is the cumulative item count its
    wrapper reports, which `Kernel.run` turns back into the actual rows.

    >>> output_dtype(int | None)
    dtype('int64')
    """
    if coded_values(annotation) is not None:
        return np.dtype(np.int64)
    item = struct_item(base_annotation(annotation))
    if item is not None:
        return struct_dtype(struct_schema(item))
    if columnar_item(base_annotation(annotation)) is not None:
        return np.dtype(np.int64)
    return numpy_dtype(base_annotation(annotation))


def literal_choices(annotation: Any) -> tuple[str, ...] | None:
    """The values of a `Literal` of strings, else `None`.

    A row node's output declared `Literal["low", "high"]` is returned by its
    compiled `fn` as the index of the value (-1 for null) and stored as the
    string; its `reference` returns the string itself.

    >>> literal_choices(typing.Literal["low", "high"]), literal_choices(str)
    (('low', 'high'), None)
    """
    if typing.get_origin(annotation) is typing.Literal and all(isinstance(a, str) for a in typing.get_args(annotation)):
        return typing.get_args(annotation)
    return None


class Lookup:
    """What `lookup(...)` declares; subclassed once per output, so it hashes by identity, never by its values."""

    values: tuple = ()
    dtype: pl.DataType = pl.Null()


def lookup(values: typing.Sequence[Any], dtype: pl.DataType) -> type[Lookup]:
    """An output annotation for a row node that returns the index of one of `values` (-1 for null).

    Like a `Literal`, for values a `Literal` can't hold: lists, records,
    lists of records. `dtype` is the polars type of the column.

    Example::

        Output("actions", lookup(([{"code": 1}], [{"code": 1}, {"code": 2}]), pl.List(pl.Struct({"code": pl.Int64}))))
    """
    return type("Lookup", (Lookup,), {"values": tuple(values), "dtype": dtype})


def coded_values(annotation: Any) -> tuple[tuple, pl.DataType] | None:
    """The values and polars type of an output a row node returns as an index, else `None`."""
    if isinstance(annotation, type) and issubclass(annotation, Lookup):
        return annotation.values, annotation.dtype
    choices = literal_choices(annotation)
    return None if choices is None else (choices, pl.String())


class CodeTable:
    """A coded output's values with null last, as Python objects and as a polars column; built once per kernel."""

    __slots__ = ("objects", "series", "mutable")

    def __init__(self, values: tuple, dtype: pl.DataType):
        # Filled one by one: `np.array` would turn a tuple of lists into a 2-d array.
        self.objects = np.empty(len(values) + 1, object)
        for k, v in enumerate(values):
            self.objects[k] = v
        self.series = pl.Series((*values, None), dtype=dtype)
        self.mutable = any(isinstance(v, (list, dict)) for v in values)


class Coded:
    """A coded output kept as each row's index into `table` (-1 for null) until it is read.

    `decode()` gives the Python values (what `run` on a record returns) and `series()`
    the polars column (what `run` on a frame returns), so neither path builds an object
    column only to convert it again.
    """

    __slots__ = ("codes", "table")

    def __init__(self, codes: np.ndarray, table: CodeTable):
        self.codes = codes
        self.table = table

    def decode(self) -> np.ndarray:
        values = self.table.objects[self.codes]
        if self.table.mutable:
            # Every row would otherwise share the table's list: a caller editing one edits them all.
            for k, v in enumerate(values):
                values[k] = copy.deepcopy(v)
        return values

    def series(self, name: str) -> pl.Series:
        return self.table.series.gather(self.codes).alias(name)


def coded(codes: np.ndarray, table: CodeTable, valid: dict[int, np.ndarray], vid: int) -> Coded:
    if (codes < 0).any():
        valid[vid] = codes >= 0
    return Coded(codes, table)
