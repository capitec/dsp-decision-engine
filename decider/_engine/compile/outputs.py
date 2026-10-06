"""How a kernel stores what a step writes."""
from __future__ import annotations

import typing
from typing import Any

import numpy as np

from decider.ops.nested import struct_item
from decider._engine.compile.njit import numpy_dtype
from decider._engine.compile.structs import struct_dtype, struct_schema
from decider._engine.ir.decls import base_annotation
from decider._engine.types import columnar_item


def output_dtype(annotation: Any) -> np.dtype:
    """The dtype a kernel stores an output declared `annotation` in: `numpy_dtype` of `T` for `T | None`.

    A `Literal` of strings is stored as the int64 index of its value; a
    `Struct[Item]` as one record per row. A `Columnar[Item]` has no
    fixed-stride column of its own: this is the cumulative item count its
    wrapper reports, which `Kernel.run` turns back into the actual rows.

    >>> output_dtype(int | None)
    dtype('int64')
    """
    if literal_choices(annotation) is not None:
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


def decode_literal(codes: np.ndarray, choices: tuple[str, ...], valid: dict[int, np.ndarray], vid: int) -> np.ndarray:
    strings = np.array((*choices, None), object)[codes]
    if (codes < 0).any():
        valid[vid] = codes >= 0
    return strings
