from __future__ import annotations

from typing import Sequence

import polars as pl

from decider.engine.ir.decls import Input, NullPolicy
from decider.exceptions import MissingInputError


def check_required(frame: pl.DataFrame, inputs: Sequence[Input], path: str = "") -> None:
    """Raise `MissingInputError` for the first REQUIRED input that is null on any row, or absent.

    Example::

        check_required(frame, [Input("income", float)], path="affordability/income")
    """
    required = [d for d in inputs if d.null_policy is NullPolicy.REQUIRED]
    if not required:
        return
    # One null_count() over the whole frame is cheaper than one per column.
    names = frame.columns
    counts = dict(zip(names, frame.null_count().row(0))) if frame.height else {}
    for decl in required:
        if decl.name not in names:
            raise MissingInputError(decl.name, path, frame.height, frame.height, absent=True)
        if counts.get(decl.name):
            raise MissingInputError(decl.name, path, counts[decl.name], frame.height)
