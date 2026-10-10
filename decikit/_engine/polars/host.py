from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping

import polars as pl

from decikit._engine.ir.context import to_ir
from decikit._engine.ir.nodes import CallNode, IRNode, SequenceNode, iter_nodes
from decikit._engine.run.engine import Executable
from decikit._engine.run.gathered import annotated_dtype, gathered_dtype, named
from decikit._engine.wiring import interface
from decikit.exceptions import EngineError, WiringError

_FRAME_NOTE = "in frame step "


@dataclass(frozen=True)
class _Segment:
    paths: tuple[str, ...]
    streamable: bool
    executable: Executable
    reads: frozenset[str]
    writes: frozenset[str]
    # Values the sub-plan outputs that no later segment and not the final output takes.
    unneeded: tuple[str, ...]
    params_paths: frozenset[str]
    shared_keys: frozenset[str]


class PolarsExecutable:
    """A pipeline bound to the polars engine: `run` a `LazyFrame` into a `LazyFrame`.

    Each run of row-local steps between frame steps runs on the fused engine
    for every batch polars hands it; each frame step sees the whole frame.
    `segments` lists them, with whether polars may stream each one, and
    `engine` is the engine document that bound it.

    Example::

        out = exe.run(pl.scan_parquet("applications.parquet"))
        out.sink_parquet("decisions.parquet")
        exe.segments   # ((("p/ratio", "p/affordable"), True), (("p/rank_income",), False), (("p/flag",), True))
    """

    def __init__(self, step_or_ir: Any, bind: Callable[[IRNode], Executable], engine: Any = None):
        self.engine = engine
        ir = step_or_ir if isinstance(step_or_ir, IRNode) else to_ir(step_or_ir)
        self.full = bind(ir)
        qualified = sorted(k for k in self.full.plan.outputs if "@" in k)
        if qualified:
            name, _, path = qualified[0].partition("@")
            raise EngineError(
                f"the polars engine outputs plain column names, and this pipeline emits {qualified[0]!r}. Relabel "
                f"the step at {path!r} so that version has its own name (`.relabel(writes={{{name!r}: "
                f"'{name}_2'}})`), or run the pipeline on the numba engine.")
        # Every name of the pipeline is declared, so a segment never mistakes a value an earlier segment
        # wrote for a typo of one it writes itself.
        declared = tuple(sorted({v.name for v in self.full.plan.versions}))
        needed: set[str] | None = set(self.full.plan.outputs)
        segments = []
        for nodes, streamable in reversed(_groups(ir)):
            writes = frozenset().union(*(interface(n)[1] for n in nodes))
            emits = writes if needed is None else writes & needed
            paths = tuple(n.origin.path for n in nodes)
            try:
                exe = bind(SequenceNode(ir.origin, nodes, emits=tuple(sorted(emits)), inputs=declared))
            except WiringError as e:
                raise EngineError(f"the polars engine runs {', '.join(paths)} as one segment, which does not wire on "
                                  f"its own: {e} Run the pipeline on the numba engine.") from e
            reads = frozenset(i.name for i in exe.plan.inputs)
            unneeded = () if needed is None else tuple(sorted(
                k for k, v in exe.plan.outputs.items() if v.producer is not None and k not in needed))
            segments.append(_Segment(
                paths, streamable, exe, reads, writes, unneeded,
                frozenset(n.path for n in exe.nodes.values()),
                frozenset(d.shared_key for n in exe.nodes.values() for d in n.decls if d.shared_key is not None)))
            # A frame step of unknown inputs may read any column, so every earlier value travels to it.
            if needed is None or any(isinstance(n, CallNode) and n.kind == "frame" and n.inputs is None
                                     for node in nodes for n in iter_nodes(node)):
                needed = None
            else:
                needed |= reads
        self._segments = tuple(reversed(segments))
        self.segments = tuple((s.paths, s.streamable) for s in self._segments)

    def run(self, data: Any, params: Mapping[str, Any] | None = None, trace: Any = None) -> Any:
        """Run a `LazyFrame` into a `LazyFrame`, a `DataFrame` into a `DataFrame`, or a dict record into a dict.

        Nothing runs until the result is collected, streamed or sunk.
        `params` is validated against the whole pipeline before the plan is
        returned, so an invalid document fails here, not at `collect()`.

        Example::

            exe.run(pl.scan_parquet("in.parquet")).collect(engine="streaming")
            exe.run(df)                                   # a DataFrame
            exe.run({"income": 1000.0, "debt": 200.0})    # a dict
        """
        if trace is not None:
            raise EngineError("the polars engine records no trace; run the pipeline on the numba engine "
                              "(`pipeline.compile('fused')`) to write a TraceSink")
        if isinstance(data, Mapping):
            return self.full.run(data, params)
        if isinstance(data, pl.DataFrame):
            return self.run(data.lazy(), params).collect()
        if not isinstance(data, pl.LazyFrame):
            raise TypeError(f"run() takes a dict record, a polars DataFrame or LazyFrame, not {type(data).__name__}")
        schema = data.collect_schema()
        # On the whole pipeline: the shadowing check, params validation and the output columns, as fused has them.
        final = _zero_rows(self.full, schema, params).columns
        lf = data
        for seg in self._segments:
            # A name this segment only writes would trip its shadowing check if an earlier segment's value stayed.
            stale = (seg.writes - seg.reads).intersection(schema.names())
            if stale:
                lf = lf.drop(sorted(stale))
                schema = pl.Schema({n: t for n, t in schema.items() if n not in stale})
            doc = _params_for(params, seg)
            out, nulls = _output_schema(seg, schema, doc)
            # Older polars let a later filter, slice or projection pass a map_batches node by default, under a
            # frame step that then sees fewer rows or columns.
            lf = lf.map_batches(partial(_run_batch, seg.executable, doc, seg.unneeded, nulls), schema=out,
                                streamable=seg.streamable, predicate_pushdown=False, projection_pushdown=False,
                                slice_pushdown=False)
            schema = out
        return lf.select(final)

    def fallbacks(self) -> dict[str, str]:
        """Every step that runs outside a compiled kernel, keyed by step path, with the reason why.

        Example::

            exe.fallbacks()   # {"order/order_total": "reads 'items' as list[dict], which no kernel takes"}
        """
        out: dict[str, str] = {}
        for seg in self._segments:
            out.update(seg.executable.fallbacks())
        return out

    def session(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported: debug the same pipeline on the numba engine, which pauses at every node."""
        raise EngineError("the polars engine has no debug session; open one on the numba engine: "
                          "`pipeline.session(df)`, or `pipeline.session(df, engine='fused')`")


def _groups(ir: IRNode) -> list[tuple[tuple[IRNode, ...], bool]]:
    # Consecutive top-level children without a frame call stream together; a child holding one sees the whole frame.
    groups: list[tuple[tuple[IRNode, ...], bool]] = []
    outside: set[str] = set()
    written: set[str] = set()
    for kid in ir.children() if isinstance(ir, SequenceNode) else (ir,):
        streamable = not any(isinstance(n, CallNode) and n.kind == "frame" for n in iter_nodes(kid))
        reads, writes = interface(kid)
        # Wired on its own, a segment reads an earlier segment's value as an input column, which no later node in
        # it may write; a row-local segment splits anywhere, so that node starts the next one.
        if streamable and groups and groups[-1][1] and not writes & (outside - written):
            groups[-1] = (groups[-1][0] + (kid,), True)
            outside |= reads - written
            written |= writes
        else:
            groups.append(((kid,), streamable))
            outside, written = set(reads), set(writes)
    return groups


def _params_for(doc: Mapping[str, Any] | None, seg: _Segment) -> dict[str, Any] | None:
    # A sub-plan rejects an entry for a step it does not hold; the whole document was validated already.
    # ponytail: a fresh document per run defeats each segment's hash-once cache; cache pruned documents if
    # small runs on many documents get slow.
    if not doc:
        return None
    out = _subtree({k: v for k, v in doc.items() if k != "shared"}, "", seg.params_paths)
    shared = doc.get("shared")
    if isinstance(shared, Mapping) and (kept := {k: v for k, v in shared.items() if k in seg.shared_keys}):
        out["shared"] = kept
    return out or None


def _subtree(level: Mapping[str, Any], prefix: str, paths: frozenset[str]) -> dict[str, Any]:
    out = {}
    for key, sub in level.items():
        path = prefix + key
        if path in paths:
            out[key] = sub
        elif isinstance(sub, Mapping) and any(p.startswith(path + "/") for p in paths):
            if kept := _subtree(sub, path + "/", paths):
                out[key] = kept
    return out


def _output_schema(seg: _Segment, schema: pl.Schema, doc: Mapping | None) -> tuple[pl.Schema, dict[str, Any]]:
    out = _zero_rows(seg.executable, schema, doc)
    if seg.unneeded:
        out = out.drop(seg.unneeded)
    # A column built from Python values (`str`, `date`) has no dtype until a row arrives; its annotation says it.
    # An each's list is checked on every batch too: one whose items do not fit their declaration keeps their fields.
    nulls = {}
    for name, dtype in out.schema.items():
        if dtype == pl.Null or name in seg.executable._gathered:
            nulls[name] = _declared(seg, name, schema)
    return pl.Schema({n: nulls.get(n, t) for n, t in out.schema.items()}), nulls


def _declared(seg: _Segment, name: str, schema: pl.Schema) -> Any:
    v = seg.executable.plan.outputs.get(name)
    gathers = seg.executable._gathers
    if v is not None and v.id in gathers:
        return gathered_dtype(gathers[v.id], gathers, {}, schema)
    annotation = None if v is None else v.annotation
    if (dtype := annotated_dtype(annotation)) is not None:
        return dtype
    # A column no version names came out of a frame step of unknown outputs.
    writer = v.producer if v is not None else ", ".join(seg.paths)
    raise EngineError(
        f"the polars engine needs every column's dtype before any row arrives, and '{name}', written by {writer} "
        f"as {named(annotation)}, has none until it holds values. Annotate the step "
        f"that writes it with a type polars can name (a TypedDict, list[float], str); a frame step declares it, "
        f"as in `writes={{{name!r}: pl.Float64}}`. Or run the pipeline on the numba engine.")


def _zero_rows(exe: Executable, schema: pl.Schema, params: Mapping | None) -> pl.DataFrame:
    try:
        return exe.run(pl.DataFrame(schema=schema), params)
    except Exception as e:
        path = next((n[len(_FRAME_NOTE):] for n in getattr(e, "__notes__", ()) if n.startswith(_FRAME_NOTE)), None)
        if path is None:
            raise
        raise EngineError(
            f"frame step {path} raised on an empty frame ({type(e).__name__}: {e}). The polars engine runs each "
            f"frame step once on zero rows to learn the columns it returns, so it must accept an empty frame.") from e


def _run_batch(exe: Executable, doc: Mapping | None, unneeded: tuple[str, ...], nulls: dict[str, Any],
               df: pl.DataFrame) -> pl.DataFrame:
    out = exe.run(df, doc)
    if unneeded:
        out = out.drop(unneeded)
    # A column `Null` on zero rows holds whatever dtype its values gave.
    fix = {c: t for c, t in nulls.items() if out.schema[c] != t}
    if not fix:
        return out
    cast = out.cast(fix, strict=False)
    for c, t in fix.items():
        # polars casts a struct by field name, dropping or nulling a field, and a float to an int by truncation.
        if not (_same_fields(out.schema[c], t) and cast[c].cast(out.schema[c], strict=False).equals(out[c])):
            raise EngineError(
                f"'{c}' came back as {out.schema[c]}, which its declared {t} cannot hold without changing a value "
                f"or a field, and the polars engine fixes every column's dtype before any row arrives. Return "
                f"values of the declared type, a record's fields in the declared order, or run the pipeline on "
                f"the numba engine.")
    return cast


def _same_fields(got: pl.DataType, want: pl.DataType) -> bool:
    if got == pl.Null:
        return True
    if isinstance(want, pl.Struct):
        return (isinstance(got, pl.Struct) and [f.name for f in got.fields] == [f.name for f in want.fields]
                and all(_same_fields(a.dtype, b.dtype) for a, b in zip(got.fields, want.fields)))
    if isinstance(want, (pl.List, pl.Array)):
        return isinstance(got, (pl.List, pl.Array)) and _same_fields(got.inner, want.inner)
    return (not isinstance(got, (pl.Struct, pl.List, pl.Array))
            and getattr(got, "time_zone", None) == getattr(want, "time_zone", None))
