"""`each`: run a child flow on every element of a list column, writing the enriched list back."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from decider.engine.ir.decls import Input, NullPolicy, Output
from decider.engine.ir.nodes import SubflowNode
from decider.steps.base import Step, as_step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext
    from decider.engine.wiring.plan import Plan


class EachMode(Enum):
    """How an `each` step runs its child over a list column.

    - `PER_ROW`: one child run per parent row, in Python. Fastest on a single
      `score()`, slower on a large batch.
    - `BATCH`: the list is exploded into one child frame and the child runs
      once over all items. Fast on a large batch, slower on a single record.
    """

    PER_ROW = "per_row"
    BATCH = "batch"


_PID = "__decider_each_pid__"


@dataclass(frozen=True, slots=True, eq=False)
class EachStep(Step):
    """Runs `item` on every element of `column`, adding the child's outputs as fields. Build with `each()`."""

    __module__ = "decider.steps"

    name: str
    column: str
    item: Step
    mode: EachMode
    output: str | None = None

    def to_ir(self, ctx: IRContext) -> SubflowNode:
        from decider.engine.wiring.resolve import resolve

        # The child is built under this node's own path, so its steps' origins nest under it
        # (e.g. "items/item/heavy"), which is what `Session.structure()` and `parameters()` show.
        child_ir = ctx.child(self.name).build(self.item)
        child_plan: Plan = resolve(child_ir)
        new_fields = tuple(n for n, v in child_plan.outputs.items() if v.producer is not None)
        out_name = self.output if self.output is not None else self.column
        fn = _EachRunner(child_plan, new_fields) if self.mode is EachMode.PER_ROW \
            else _BatchRunner(child_ir, self.column, out_name)
        # A null list reads as a row with no items, the same in both modes. `arg="items"` matches
        # the child functions' own parameter name, independent of the column's name.
        inp = Input(self.column, list[dict], NullPolicy.MISSING_AS, [], arg="items")
        out = Output(out_name, list[dict])
        kind = "scalar" if self.mode is EachMode.PER_ROW else "frame"
        # Empty `params`: the child's params live under `subflow` and are surfaced by
        # `iter_with_subflows`; declaring them here would list each one twice in `parameters()`.
        return SubflowNode(ctx.origin(self), kind, fn, (inp,), (out,), (), subflow=child_ir)


class _EachRunner:
    """Runs a PER_ROW each node's child once per item; `runner` is swappable so a Session can step into it."""

    def __init__(self, plan: Plan, new_fields: tuple[str, ...]):
        # Drives the child plan directly rather than nesting a whole `Engine`: `each` already knows
        # exactly which fields it wants back (`new_fields`), so it skips the frame/shadowing/output
        # bookkeeping `Engine.score` carries for an arbitrary top-level pipeline, and pays for a
        # `RunParams` once per parent row instead of once per item. `FusedRunner` compiles the child
        # plan into numba kernels on its first `iterate()` call (cached on its own `_plan`, same as
        # `Engine.bind(..., mode="fused")` does), so every item after the first calls compiled code,
        # not a per-node Python walk; a step that can't compile still runs, one call per row.
        # The child's params arrive as `doc`, the parent's whole params document, unmodified: the child
        # was built under this node's path, so its nodes read their params from their own nested keys.
        import polars as pl

        from decider.engine.ir.decls import base_annotation
        from decider.engine.params import NodeParams, ParamsCache
        from decider.engine.run.runners.fused import FusedRunner
        from decider.engine.run.state import dtype_of

        self.plan = plan
        self.new_fields = new_fields
        self.runner = FusedRunner()
        self.nodes = {c.id: NodeParams(c.node.origin.path, c.node.params) for c in plan.calls if c.node.params}
        self.cache = ParamsCache()
        grouped = {}
        for v in plan.versions:
            if v.producer is None:
                grouped.setdefault(dtype_of(base_annotation(v.annotation)), []).append(v)
        self.inputs = list(grouped.items())
        self.results = [(f, plan.outputs[f]) for f in new_fields]
        self.empty_frame = pl.DataFrame()

    def __call__(self, items, doc):
        from decider.engine.run.params import RunParams
        from decider.engine.run.state import State, load_record, record_value

        params = RunParams(self.nodes, doc, self.cache, lazy=False)
        out = []
        for item in items or ():
            state = State(self.plan, self.empty_frame, 1)
            for dtype, versions in self.inputs:
                load_record(state, item, versions, dtype)
            for _ in self.runner.iterate(self.plan, state, params):
                pass
            row = {}
            for f, v in self.results:
                values, valid = state.read(v)
                row[f] = None if valid is not None and not valid[0] else record_value(values.tolist()[0], v.annotation)
            out.append({**item, **row})
        return out

    def child_run(self, item, doc, runner):
        """A `(state, params, checkpoints)` triple for one `item`, run with `runner`; for a stepping `Session`.

        The child is driven a checkpoint at a time by whichever thread drains `checkpoints`, so a
        session can pause it without the fused `self.runner` used for the real run.
        """
        from decider.engine.run.params import RunParams
        from decider.engine.run.state import State, load_record

        params = RunParams(self.nodes, doc, self.cache, lazy=False)
        state = State(self.plan, self.empty_frame, 1)
        for dtype, versions in self.inputs:
            load_record(state, item, versions, dtype)
        return state, params, runner.iterate(self.plan, state, params)


class _BatchRunner:
    """Runs a BATCH each node's child over the exploded frame; `exe` is swappable so a Session can rebind it."""

    def __init__(self, child_ir, column: str, out_name: str):
        self.child_ir = child_ir
        self.column = column
        self.out_name = out_name
        self.exe = None

    def __call__(self, df, doc):
        import polars as pl

        from decider.engine import Engine

        if self.exe is None:
            self.exe = Engine().bind(self.child_ir, mode="fused")
        dtype = df.schema[self.column]
        if not (isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct)):
            # Not List(Struct): either all rows are empty/null (correct) or the wrong type.
            # prepare() may have cast a fully-null list column to List(String) or String.
            is_list = isinstance(dtype, pl.List)
            has_items = (df[self.column].list.len().fill_null(0).sum() > 0 if is_list
                         else df[self.column].is_not_null().any())
            if has_items:
                raise ValueError(f"each({self.column!r}): batch mode requires a List(Struct) column, got {dtype}")
            return df.with_columns(pl.Series(self.out_name, [[]] * df.height, dtype=pl.List(pl.Null)))
        idx = df.with_row_index(_PID)
        nonzero = idx.filter(pl.col(self.column).list.len().fill_null(0) > 0)
        exploded = nonzero.explode(self.column).unnest(self.column)
        result = self.exe.run(exploded, params=doc)
        pass_through = set(df.columns) - {self.column}
        names = [c for c in result.columns if c != _PID and c not in pass_through]
        grouped = (result.select(_PID, pl.struct(names).alias(self.out_name))
                   .group_by(_PID, maintain_order=True).agg(pl.col(self.out_name)))
        joined = idx.select(_PID).join(grouped, on=_PID, how="left")
        # Null and empty lists explode to zero rows and rejoin as null: read them as no items.
        return df.with_columns(joined.get_column(self.out_name).fill_null([]).alias(self.out_name))


def each(column: str, item: Any, *, name: str | None = None,
         execution_mode: EachMode = EachMode.PER_ROW, output: str | None = None) -> EachStep:
    """Run `item` on every element of the list column `column`, writing the enriched list back.

    `item` is a flow of ordinary steps reading the item's fields; its outputs
    are added as new fields of each item. The list is read under `column` and,
    by default, written back to it in place.

    Args:
        execution_mode: `EachMode.PER_ROW` (default) runs the child once per
            parent row, in Python, which is fastest on a single `score()` and
            slower on a large batch. `EachMode.BATCH` explodes the list into one
            child frame and runs the child once over all items, which is fastest
            on a batch and slower on a single record.
        output: the column to write the enriched list to (default: `column`,
            overwriting it). Set this to keep the original list untouched
            alongside the enriched one.

    Example::

        def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
            return weight > heavy_kg

        pipeline = flow(each("items", flow(heavy, name="item"), name="items"), name="order")
    """
    return EachStep(column if name is None else name, column, as_step(item), execution_mode, output)
