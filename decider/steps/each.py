"""`each`: run a child flow on every element of a list column, writing the enriched list back."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from decider.engine.ir.decls import Input, NullPolicy, Output
from decider.engine.ir.nodes import CallNode
from decider.exceptions import IRError
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

    def to_ir(self, ctx: IRContext) -> CallNode:
        from decider.engine.ir.context import IRContext as _Ctx
        from decider.engine.wiring.resolve import resolve

        # The child is its own plan: its inputs are the item's fields, its params read from a
        # relative document the each node forwards. Built under a fresh root so its paths stay
        # relative to itself.
        child_ir = _Ctx().build(self.item)
        child_plan: Plan = resolve(child_ir)
        new_fields = tuple(n for n, v in child_plan.outputs.items() if v.producer is not None)
        params, paths = _hoist(child_plan, self.name)
        out_name = self.output if self.output is not None else self.column
        fn = _per_row(child_plan, new_fields, paths) if self.mode is EachMode.PER_ROW \
            else _batch(child_ir, self.column, out_name, paths)
        # A null list reads as a row with no items, the same in both modes. `arg="items"` matches
        # the child functions' own parameter name, independent of the column's name.
        inp = Input(self.column, list[dict], NullPolicy.MISSING_AS, [], arg="items")
        out = Output(out_name, list[dict])
        kind = "scalar" if self.mode is EachMode.PER_ROW else "frame"
        return CallNode(ctx.origin(self), kind, fn, (inp,), (out,), params)


def _hoist(plan: Plan, name: str) -> tuple[tuple, dict[str, tuple[str, str]]]:
    # The child's params become the each node's params, so `parameters()` lists them under the
    # each node's path and a retuned value reaches the child. `paths` maps each param's argument
    # to the child node path and param name it belongs to, so the forwarder rebuilds the child doc.
    params, paths = [], {}
    for call in plan.calls:
        for d in call.node.params:
            if d.shared_key is not None:
                raise IRError(f"each {name!r}: shared param {d.shared_key!r} in the child flow is not supported")
            if d.arg in paths:
                raise IRError(f"each {name!r}: two child steps have a param fed by argument {d.arg!r}; rename one")
            paths[d.arg] = (call.node.origin.path, d.name)
            params.append(d)
    return tuple(params), paths


def _child_doc(child_params: dict, paths: dict[str, tuple[str, str]]) -> dict:
    doc: dict = {}
    for arg, value in child_params.items():
        path, param = paths[arg]
        target = doc
        for part in path.split("/"):
            target = target.setdefault(part, {})
        target[param] = value
    return doc


def _per_row(plan: Plan, new_fields: tuple[str, ...], paths):
    # Drives the child plan directly rather than nesting a whole `Engine`: `each` already knows
    # exactly which fields it wants back (`new_fields`), so it skips the frame/shadowing/output
    # bookkeeping `Engine.score` carries for an arbitrary top-level pipeline, and pays for a
    # `RunParams` once per parent row instead of once per item. `FusedRunner` compiles the child
    # plan into numba kernels on its first `iterate()` call (cached on its own `_plan`, same as
    # `Engine.bind(..., mode="fused")` does), so every item after the first calls compiled code,
    # not a per-node Python walk; a step that can't compile still runs, one call per row.
    import polars as pl

    from decider.engine.ir.decls import base_annotation
    from decider.engine.params import NodeParams, ParamsCache
    from decider.engine.run.params import RunParams, check_namespaces
    from decider.engine.run.runners.fused import FusedRunner
    from decider.engine.run.state import State, dtype_of, load_record, record_value

    runner = FusedRunner()
    nodes = {c.id: NodeParams(c.node.origin.path, c.node.params) for c in plan.calls if c.node.params}
    cache = ParamsCache()
    grouped = {}
    for v in plan.versions:
        if v.producer is None:
            grouped.setdefault(dtype_of(base_annotation(v.annotation)), []).append(v)
    inputs = list(grouped.items())
    results = [(f, plan.outputs[f]) for f in new_fields]
    empty_frame = pl.DataFrame()

    def run(items, **child_params):
        doc = _child_doc(child_params, paths)
        check_namespaces(doc, nodes)
        params = RunParams(nodes, doc, cache, lazy=False)
        out = []
        for item in items or ():
            state = State(plan, empty_frame, 1)
            for dtype, versions in inputs:
                load_record(state, item, versions, dtype)
            for _ in runner.iterate(plan, state, params):
                pass
            row = {}
            for f, v in results:
                values, valid = state.read(v)
                row[f] = None if valid is not None and not valid[0] else record_value(values.tolist()[0], v.annotation)
            out.append({**item, **row})
        return out

    return run


def _batch(child_ir, column: str, out_name: str, paths):
    import polars as pl

    from decider.engine import Engine

    exe = None

    def run(df, **child_params):
        nonlocal exe
        if exe is None:
            exe = Engine().bind(child_ir, mode="fused")
        dtype = df.schema[column]
        if not (isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct)):
            # Not List(Struct): either all rows are empty/null (correct) or the wrong type.
            # prepare() may have cast a fully-null list column to List(String) or String.
            is_list = isinstance(dtype, pl.List)
            has_items = (df[column].list.len().fill_null(0).sum() > 0 if is_list
                         else df[column].is_not_null().any())
            if has_items:
                raise ValueError(f"each({column!r}): batch mode requires a List(Struct) column, got {dtype}")
            return df.with_columns(pl.Series(out_name, [[]] * df.height, dtype=pl.List(pl.Null)))
        doc = _child_doc(child_params, paths)
        idx = df.with_row_index(_PID)
        nonzero = idx.filter(pl.col(column).list.len().fill_null(0) > 0)
        exploded = nonzero.explode(column).unnest(column)
        result = exe.run(exploded, params=doc)
        pass_through = set(df.columns) - {column}
        names = [c for c in result.columns if c != _PID and c not in pass_through]
        grouped = (result.select(_PID, pl.struct(names).alias(out_name))
                   .group_by(_PID, maintain_order=True).agg(pl.col(out_name)))
        joined = idx.select(_PID).join(grouped, on=_PID, how="left")
        # Null and empty lists explode to zero rows and rejoin as null: read them as no items.
        return df.with_columns(joined.get_column(out_name).fill_null([]).alias(out_name))

    return run


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
