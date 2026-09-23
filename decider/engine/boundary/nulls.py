from __future__ import annotations

from typing import Sequence

import polars as pl

from decider.engine.ir.decls import Input, NullPolicy


class MissingInputError(ValueError):
    """A required input (no `missing_as()`, no `| None`) is null or absent in the data.

    `input`, `path` and `null_count` say which input of which step, and on how many rows.
    """

    def __init__(self, input: str, path: str, null_count: int, n_rows: int, absent: bool = False):
        self.input, self.path, self.null_count = input, path, null_count
        where = f"step '{path}'" if path else "the pipeline"
        found = ("is not in the input frame" if absent
                 else f"has {null_count} null row(s) of {n_rows}")
        super().__init__(
            f"input '{input}' of {where} is required but column '{input}' {found}. "
            f"Fix the data, or declare `{input}: T = missing_as(fill)` or `{input}: T | None`."
        )


def check_required(frame: pl.DataFrame, inputs: Sequence[Input], path: str = "") -> None:
    """Raise `MissingInputError` for the first REQUIRED input that is null on any row, or absent.

    Example::

        check_required(frame, [Input("income", float)], path="affordability/income")
    """
    required = [d for d in inputs if d.null_policy is NullPolicy.REQUIRED]
    if not required:
        return
    # One null_count() over the whole frame is cheaper than one per column.
    counts = dict(zip(frame.columns, frame.null_count().row(0))) if frame.height else {}
    for decl in required:
        if decl.name not in frame.columns:
            raise MissingInputError(decl.name, path, frame.height, frame.height, absent=True)
        if counts.get(decl.name):
            raise MissingInputError(decl.name, path, counts[decl.name], frame.height)
