"""A decision-table document as a pipeline element (doc 03 §5.3).

Like `decider2.trees.build`, this returns an ordinary `types.Module` — the
pipeline learns nothing new. The one extra obligation a table has and a tree
does not is `shared`: its rows live in numpy arrays passed to `apply`/
`score` through the reserved `shared` bundle (doc 03 §4.2), which is exactly
what makes a row edit free of compilation (doc 08 §3.4).

    dt = table_module(DecisionTable.model_validate(doc))
    out = flow(dt.module).apply(frame, shared=dt.shared)

`dt.shared` is a plain dict, so several tables compose by merging:
`shared={**a.shared, **b.shared}`. Names are prefixed with the table's
instance name, so two tables never collide.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from decider2.compile import cache
from decider2.graph.module import module
from decider2.tables.codegen import EmittedTable, emit_table
from decider2.tables.schema import DecisionTable
from decider2.trees.build import DEFAULT_BUILD_DIR
from decider2.types import Module

__all__ = ["TableModule", "table_module"]


@dataclass(frozen=True)
class TableModule:
    """A built decision table: its `Module`, its `shared` arrays, its report."""

    module: Module
    table: DecisionTable
    emitted: EmittedTable
    source_path: Path

    @property
    def name(self) -> str:
        return self.module.name

    @property
    def shared(self) -> dict[str, np.ndarray]:
        """The table's rows, as the arrays the kernel reads.

        Rebuild this (via `table_module`) and the answers change with no
        compilation — the property doc 08 §3.4 calls a free interior change.
        """
        return dict(self.emitted.shared)

    @property
    def row_column(self) -> str:
        """The column naming which row matched — the table's path capture,
        -1 when nothing matched. Emit it like any other value."""
        return self.emitted.row_fn_name

    def decode(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the string-valued output columns back.

        Same reason as `trees.build.TreeModule.decode`: a kernel writes
        float64/int64/bool arrays only, so a string output is mapped from
        the matched row index in polars instead.
        """
        row_col = self.row_column
        if row_col not in frame.columns:
            raise KeyError(
                f"decode() needs the '{row_col}' column, which carries which row "
                f"matched. Pass it through, or add .emit('{row_col}')."
            )
        exprs = []
        for column in self.table.outputs:
            values = self.table.parameters.column(column)
            if not any(isinstance(v, str) for v in values):
                continue
            mapping = {i: v for i, v in enumerate(values)}
            exprs.append(
                pl.col(row_col)
                .replace_strict(
                    mapping,
                    default=self.table.default_for(column),
                    return_dtype=pl.String,
                )
                .alias(column)
            )
        return frame.with_columns(exprs) if exprs else frame

    def explain(self) -> str:
        e = self.emitted
        return "\n".join(
            [
                f"decision table {self.table.name!r} -> module {self.module.name!r}",
                f"  rows            : {e.n_rows} (arrays in shared — editing them is free)",
                f"  conditions      : {e.n_conditions} (shape — editing these recompiles)",
                f"  emitted lines   : {e.emitted_lines} (cap 500, doc 05 §7)",
                f"  variables       : {', '.join(e.variables) or '(none)'}",
                f"  string variables: {', '.join(e.string_variables) or '(none)'}",
                f"  row column      : {self.row_column}",
                f"  output steps    : {', '.join(e.output_fn_names) or '(none)'}",
                f"  shared keys     : {len(e.shared)}",
                f"  source          : {self.source_path}",
            ]
        )


def table_module(
    table: DecisionTable,
    *,
    name: str | None = None,
    build_dir: "str | Path | None" = None,
) -> TableModule:
    """Compile a decision-table document into a pipeline element."""
    emitted = emit_table(table, name=name)
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    cached = cache.get_or_build(emitted.source, build_dir)

    fns = [getattr(cached.module, fn) for fn in emitted.matcher_fn_names]
    fns.append(getattr(cached.module, emitted.row_fn_name))
    fns += [getattr(cached.module, fn) for fn in emitted.output_fn_names]

    built = module(*fns, name=name or table.name or "decision_table")
    return TableModule(
        module=built, table=table, emitted=emitted, source_path=Path(cached.path)
    )
