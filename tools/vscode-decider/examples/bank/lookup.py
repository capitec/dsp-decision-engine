"""Lookup tables as steps: product, rate and fee tables whose rows are params.

A row matches when every `keys` column equals the record's value and every
`ranges` column pair holds it; the first matching row wins. The rows are a
param, so they can be edited in a params document, and the step also writes
`<name>_row`, the index of the row that matched (-1 for none).

Example::

    rates = LookupTable(
        name="pl_base_rates",
        ranges={"requested_term": ["min_term", "max_term"]},
        returns={"pl_base_rate": "rate"},
        rows=[{"min_term": 6, "max_term": 24, "rate": 0.215}],
    )
"""
from __future__ import annotations

import typing as t

from decider.engine.ir.decls import Input, Output, ParamDecl
from decider.engine.ir.nodes import CallNode
from decider.steps import ConfigurableStep


class LookupTable(ConfigurableStep):
    keys: t.Dict[str, str] = {}
    ranges: t.Dict[str, t.List[str]] = {}
    returns: t.Dict[str, str]
    rows: t.List[t.Dict[str, t.Any]]

    def to_ir(self, ctx: t.Any) -> CallNode:
        keys, ranges, returns = dict(self.keys), {k: tuple(v) for k, v in self.ranges.items()}, dict(self.returns)

        def lookup(rows: list, **record: t.Any) -> tuple:
            for i, row in enumerate(rows):
                if all(record[k] == row[c] for k, c in keys.items()) and all(row[lo] <= record[k] <= row[hi] for k, (lo, hi) in ranges.items()):
                    return (*(row[c] for c in returns.values()), i)
            return (*(None for _ in returns), -1)

        lookup.__qualname__ = lookup.__name__ = f"lookup_{self.name}"
        inputs = tuple(Input(k, t.Any) for k in [*keys, *ranges])
        outputs = (*(Output(o, float) for o in returns), Output(f"{self.name}_row", int))
        return CallNode(ctx.origin(self), "scalar", lookup, inputs, outputs, (ParamDecl("rows", list, list(self.rows)),))
