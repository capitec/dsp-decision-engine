"""Decision tables as data-shaped module interiors (doc 08 §3, §3.4).

    from decider2 import flow
    from decider2.tables import DecisionTable, table_module

    dt = table_module(DecisionTable.model_validate(doc))
    out = flow(dt.module).apply(frame, shared=dt.shared)
    out = dt.decode(out)          # string-valued output columns

A table is a pydantic document (`schema.py`), emitted as ONE generic
row-scan kernel (`codegen.py`) whose rows live in `shared` arrays, and
wrapped as an ordinary `Module` (`build.py`). Doc 08 §3.4 chooses this shape
over codegen, and its consequence is the headline property: **editing a
table's rows is free — no compile, not even a staged one.**

MIGRATION NOTES — what did not come across from decider 1, and why:

* **`output_fn`** (`impl.calculate_decision_table_output`'s injectable
  `default_form_output_struct_from_row`): a callable in config. Doc 08 §1.1
  forbids it — "config may reference code by registered id. It may not
  contain code". The output shape is the declared `outputs` list instead.
* **`unnest_output`**: decider 1 returned a struct column and needed a flag
  to flatten it for downstream steps. decider2's output frame is flat by
  construction (doc 03 §7), so every output column is already top level.
* **String `in`/`eq` literals are shape, not data.** For a numeric table
  every row edit is free. For a string column, retuning a literal's *text*
  is free (it is a `str` param, resolved to an int32 code — EXPERIMENTS.md
  §O), but introducing a *new distinct literal* changes the hoisted
  matcher's signature and recompiles. See `codegen.py`'s docstring.
* **Regex/`contains`-style matching** never existed in decider 1's table
  vocabulary and is not added; doc 05 §1.5 puts that work in the frame tier.
"""
from __future__ import annotations

from decider2.tables.build import TableModule, table_module
from decider2.tables.encode import EncodedTable, TableTooComplex, encode_table
from decider2.tables.schema import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    DecisionTable,
    EqExpression,
    Expression,
    InExpression,
    IsTrueExpression,
    OrExpression,
    ParametersConfig,
)

__all__ = [
    "DecisionTable",
    "table_module",
    "TableModule",
    "encode_table",
    "EncodedTable",
    "TableTooComplex",
    "ParametersConfig",
    "BoundMode",
    "AndExpression",
    "OrExpression",
    "BetweenExpression",
    "InExpression",
    "IsTrueExpression",
    "EqExpression",
    "Expression",
]
