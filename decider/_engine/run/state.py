from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any, Hashable, Mapping

import numpy as np
import polars as pl

from decider._engine.compile.outputs import Coded, coded_values
from decider._engine.ir.absent import has_optional_key, without_absent
from decider._engine.ir.decls import Input, NullPolicy, base_annotation, is_typed
from decider._engine.run.representations import despan
from decider._engine.wiring.plan import Plan, NodeVersion
from decider.ops.nested import struct_item
from decider._engine.types import annotation_cache, columnar_item, decimal_info, is_decimal, item_schema, to_raw_input


class State:
    """Every value of one run: a numpy array per `NodeVersion.id`, a validity mask for nullable ones.

    Args:
        values: `NodeVersion.id` -> array of `n` values, or a `Coded` `Literal`
            output, decoded when read.
        valid: `NodeVersion.id` -> bool mask of `n` rows; absent means every row is valid.
        frame: the frame unread columns pass through from: the input frame,
            replaced by what an unknown-lineage frame step returns.

    Example::

        state = executable.prepare(df)[0]
        state.column("term_cap@term/cap_by_income")   # a polars Series
        state.versions("term_cap@*")                  # every version, in production order
    """

    def __init__(self, plan: Plan, frame: pl.DataFrame, n: int | None = None):
        self.plan = plan
        self.frame = frame
        self.n = frame.height if n is None else n
        self.values: dict[int, np.ndarray | Coded] = {}
        self.valid: dict[int, np.ndarray] = {}
        self.chains: dict[str, list[NodeVersion]] = {k: list(v) for k, v in plan.chains.items()}
        self._extra = 0
        self._sources: dict[int, tuple[np.ndarray, pl.Series]] = {}
        # Per version id: the compiled representations built from its values, each with the
        # buffers it borrows. Dropped whenever those values are written again.
        self._representations: dict[int, dict[Hashable, tuple[np.ndarray, list]]] = {}

    @classmethod
    def from_frame(cls, plan: Plan, frame: pl.DataFrame, n: int | None = None) -> State:
        """A state holding the plan's input columns read from `frame`; absent columns are all null."""
        from decider._engine.boundary.extract import extract_frame

        state = cls(plan, frame, n)
        names = frame.columns
        typed = [i for i in _typed(plan) if i.name in names]
        extracted = extract_frame(frame, typed).columns if typed else {}
        for v in plan.versions:
            if v.producer is not None:
                continue
            col = extracted.get(v.name)
            if col is not None:
                state.write(v, col.values, valid=col.validity if col.has_nulls else None)
            elif v.name in names:
                # ponytail: a string column becomes Python objects even when only kernels read it
                # (as spans, from `source`); convert on first Python read if big string batches matter.
                series = frame.get_column(v.name)
                values, valid = from_series(series)
                if has_optional_key(v.annotation):
                    for i, x in enumerate(values):
                        values[i] = without_absent(x, v.annotation)
                state.write(v, values, valid=valid)
                state._sources[v.id] = (values, series)
        return state

    def write(self, version: NodeVersion, values: np.ndarray, rows: np.ndarray | None = None,
              valid: np.ndarray | None = None) -> None:
        """Store `values` for `version` on `rows`; `valid` marks which of them are non-null.

        With `rows=None` the array is stored as is (not copied). Rows a write
        never touches stay null.
        """
        vid = version.id
        self._representations.pop(vid, None)   # built from the values this write replaces
        if rows is None:
            self.values[vid] = values
            if valid is not None and not valid.all():
                self.valid[vid] = valid
            else:
                self.valid.pop(vid, None)
            return
        if isinstance(values, Coded):
            values = values.decode()
        target = self.values.get(vid)
        if isinstance(target, Coded):
            target = self.values[vid] = target.decode()
        if target is None:
            target = self.values[vid] = np.full(self.n, None, object) if values.dtype == object else np.zeros(self.n, values.dtype)
            self.valid[vid] = np.zeros(self.n, bool)
        elif values.dtype == object and target.dtype != object:
            target = self.values[vid] = target.astype(object)
        elif values.dtype != object and target.dtype == object and values.dtype.names is not None:
            # A struct carry seeded null (object) is later written as a record; rows still null stay zero records.
            target = self.values[vid] = np.zeros(self.n, values.dtype)
        target[rows] = values
        mask = self.valid.get(vid)
        if mask is None:
            if valid is None or valid.all():
                return
            mask = self.valid[vid] = np.ones(self.n, bool)
        mask[rows] = True if valid is None else valid

    def read(self, version: NodeVersion, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """`version`'s values and validity mask (`None` = all valid) on `rows`; never written means all null."""
        values, valid = self.values.get(version.id), self.valid.get(version.id)
        if values is None:
            values, valid = np.full(self.n, None, object), np.zeros(self.n, bool)
        elif isinstance(values, Coded):
            values = values.decode()
        if rows is None:
            return values, valid
        return values[rows], None if valid is None else valid[rows]

    def source(self, version: NodeVersion) -> pl.Series | None:
        """The input frame column `version` was read from, while its values are still that column's.

        Lets a compiled step read an object column's Arrow buffers in place.

        Example::

            state.source(state.versions("channel")[0])   # the frame's "channel" Series
        """
        values, series = self._sources.get(version.id, (None, None))
        return series if values is not None and self.values.get(version.id) is values else None

    def representation(self, version: NodeVersion, kind: Hashable, build) -> np.ndarray:
        """Return cached compiled representation for `version` and `kind`.

        `build(values, source, alive)` receives full-column values and may append
        backing buffers to `alive` when its representation borrows memory.
        """
        cached = self._representations.setdefault(version.id, {})
        entry = cached.get(kind)
        if entry is None:
            alive: list = []
            values = self.read(version)[0]
            entry = cached[kind] = (build(values, self.source(version), alive), alive)
        return entry[0]

    def record(self, name: str, producer: str, values: np.ndarray, valid: np.ndarray | None = None) -> NodeVersion:
        """Store `values` as a new version of `name`, appended to its chain; for overrides.

        Example::

            state.record("disposable_income", "override@term/cap_by_income", np.full(state.n, 1200.0))
        """
        self._extra += 1
        v = NodeVersion(len(self.plan.versions) + self._extra - 1, name, producer)
        self.chains.setdefault(name, []).append(v)
        self.write(v, values, valid=valid)
        return v

    def restore(self, old: State, keep: set[int]) -> None:
        """Take back from `old` the values of the versions in `keep`, every override it recorded, and its chains.

        For re-running part of a run: a fresh state replays up to a node, then
        restores what the earlier run had upstream of it.

        Example::

            fresh.restore(state, {v.id for v in plan.versions if v.producer is None})
        """
        for vid, values in old.values.items():
            if vid in keep or vid >= len(self.plan.versions):
                self.values[vid] = values
                if vid in old.valid:
                    self.valid[vid] = old.valid[vid]
                else:
                    self.valid.pop(vid, None)
        self.chains, self._extra = old.chains, old._extra
        self._representations = {}   # values this rewinds past may have built them

    def versions(self, spec: str) -> list[NodeVersion]:
        """The versions `spec` names: `name` (the latest), `name@path` (by producer) or `name@*` (all).

        Example::

            [v.producer for v in state.versions("term_cap@*")]
        """
        name, at, producer = spec.partition("@")
        chain = self.chains.get(name, [])
        if producer == "*":
            return list(chain)
        if at:
            return [v for v in chain if v.producer == producer][-1:]
        if chain:
            return chain[-1:]
        return [v for v in self.plan.versions if v.producer is None and v.name == name]

    def column(self, spec: str, version: NodeVersion | None = None) -> pl.Series:
        """One version as a polars Series named `spec`, null where invalid."""
        (v,) = [version] if version is not None else self.versions(spec)
        source = self.source(v)
        if source is not None:
            return source if source.name == spec else source.alias(spec)
        values = self.values.get(v.id)
        if isinstance(values, Coded):
            return values.series(spec)
        return _series(spec, *self.read(v), v.annotation)

    def frame_of(self, base: pl.DataFrame, names: dict[str, NodeVersion], rows: np.ndarray | None) -> pl.DataFrame:
        """`base` with the current value of every name in `names` on `rows`, for a frame step."""
        cols = []
        for name, v in names.items():
            source = self.source(v)
            if source is not None:
                cols.append(source if rows is None else source.gather(rows))
            else:
                cols.append(_series(name, *self.read(v, rows), v.annotation))
        return base.with_columns(cols) if cols else base


# Keyed by the plan's identity; a fill value may not be hashable.
@lru_cache(maxsize=64)
def _typed(plan: Plan) -> tuple[Input, ...]:
    # The inputs the boundary reads into typed arrays, nulls kept as a mask.
    return tuple(Input(i.name, i.annotation if decimal_info(i.annotation) else t, NullPolicy.OPTIONAL)
                 for i in plan.inputs if is_typed(t := base_annotation(i.annotation)))


def _series(name: str, values: np.ndarray, valid: np.ndarray | None, annotation: Any = None) -> pl.Series:
    if values.dtype == object:
        rows = (values if valid is None else np.where(valid, values, None))
        # A struct output read back into Python is a numpy record or a tuple; polars reads dicts (not
        # tuples) for a Struct with nulls, so spell it as the dict a step returns.
        values = [record_value(x, annotation) for x in rows.tolist()]
        # Polars infers a list's element type from its first row: `[[1, 2], [1.5]]` would become ints,
        # and an all-null `Literal` column `Null`.
        dtype = declared_dtype(annotation)
        keep = isinstance(dtype, (pl.List, pl.Struct, pl.Array)) or coded_values(base_annotation(annotation))
        if keep:
            return pl.Series(name, values, dtype=dtype)
        try:
            return pl.Series(name, values)
        except TypeError:
            # A step may return an int in one row and a float in the next, or a dict with int keys,
            # which `score` hands back as is; polars then widens the type, or holds the dicts as objects.
            return pl.Series(name, values, strict=False)
    if values.dtype.names is not None:
        # A struct output is a record per row; polars reads dicts (not tuples) for a Struct
        # with nulls, so spell each row as the dict a step returns.
        names = values.dtype.names
        rows = [None if valid is not None and not valid[i] else dict(zip(names, row))
                for i, row in enumerate(values.tolist())]
        dtype = declared_dtype(annotation)
        return pl.Series(name, rows, dtype=dtype if isinstance(dtype, pl.Struct) else None)
    s = pl.Series(name, values)
    if valid is not None:
        s = s.scatter(np.flatnonzero(~valid), None)
    if (info := decimal_info(annotation)) is not None:
        # The int64 mantissa as a Decimal(38, scale): cast scales it up, the division brings it back.
        s = s.cast(pl.Decimal(38, info.scale)) / 10 ** info.scale
    return s


def record_value(value: Any, annotation: Any = None) -> Any:
    """A single record row as the dict a step returns, from `values.tolist()[i]`.

    A compiled struct output is a numpy record, which `tolist()` turns into a
    tuple of fields (or leaves as a `np.void` in an object array); a step
    returns a dict, so `score` must agree. A Decimal value is boxed by the caller.
    """
    item = struct_item(base_annotation(annotation))
    if item is not None and value is not None and not isinstance(value, dict):
        return dict(zip((name for name, _ in item_schema(item)),
                        value.tolist() if isinstance(value, np.void) else value))
    return despan(value)


def load_record(state: State, record: Mapping[str, Any], versions: list[NodeVersion], dtype: np.dtype) -> None:
    """Write `record`'s value of every one-row `versions` `dtype` groups into `state`.

    One array per dtype, viewed per input, is far cheaper than one array per
    input; shared by `Engine.score`'s record path and `each`'s per-item runs,
    so both score a plain dict the same way.
    """
    given = [v for v in versions if v.name in record]
    if not given:
        return
    xs = [record[v.name] for v in given]
    if dtype == np.int64:
        xs = [x if (info := decimal_info(v.annotation)) is None or x is None else to_raw_input(x, info.scale)
              for x, v in zip(xs, given)]
    nulls = [x is None for x in xs]
    if dtype == object:
        block = np.empty(len(xs), object)
        # One by one, so a list or tuple value stays one element.
        for k, x in enumerate(xs):
            block[k] = x
    else:
        block = np.array([0 if null else x for x, null in zip(xs, nulls)] if any(nulls) else xs, dtype)
        # Read-only like a column read from a frame, so one kernel specialisation serves both.
        block.flags.writeable = False
    for k, (v, null) in enumerate(zip(given, nulls)):
        state.write(v, block[k:k + 1], valid=np.zeros(1, bool) if null else None)


def from_series(s: pl.Series) -> tuple[np.ndarray, np.ndarray | None]:
    """A polars Series as `(values, valid)`; nulls hold a zero (or `None` for objects) and are masked."""
    if s.dtype.is_nested() or s.dtype.is_temporal():
        # numpy turns structs into float rows, dates into epoch ints and lists into arrays; keep Python objects.
        values = np.empty(len(s), object)
        for i, x in enumerate(s.to_list()):
            values[i] = x
        return values, (s.is_not_null().to_numpy() if s.null_count() else None)
    if not s.null_count():
        return s.to_numpy(), None
    valid = s.is_not_null().to_numpy()
    if s.dtype.is_numeric():
        s = s.fill_null(0)
    elif s.dtype == pl.Boolean:
        s = s.fill_null(False)
    return s.to_numpy(), valid


def fill_missing(values: np.ndarray, valid: np.ndarray, fill: Any) -> np.ndarray:
    """`values` with `fill` on every row `valid` marks missing.

    A list, dict or tuple fill is one value per row (a fresh copy each, so a
    step mutating it changes neither the declaration nor other rows);
    `np.where` would broadcast it into zero or several columns.
    """
    if not isinstance(fill, (list, dict, tuple)):
        return np.where(valid, values, fill)
    out = values.astype(object)
    for i in np.flatnonzero(~valid):
        out[i] = copy.deepcopy(fill)
    return out


@annotation_cache
def declared_dtype(annotation: Any) -> pl.DataType | None:
    """The polars dtype of `annotation` (`list[float]` -> `List(Float64)`), or `None` when polars can't say.

    A `Struct[Item]` is a `Struct` of its fields, a `Columnar[Item]` a
    `List` of it, so an all-empty such column still keeps its declared shape
    rather than inferring `Null`.
    """
    if annotation is None:
        return None
    if (values := coded_values(base_annotation(annotation))) is not None:
        return values[1]
    item = struct_item(base_annotation(annotation))
    if item is not None:
        return _item_dtype(item)
    item = columnar_item(annotation)
    if item is not None:
        return pl.List(_item_dtype(item))
    try:
        dtype = pl.DataType.from_python(base_annotation(annotation))
    except (TypeError, ValueError):
        return None
    inner = dtype
    while isinstance(inner, (pl.List, pl.Array)):
        inner = inner.inner
    # A bare `list` has no element type (polars gives the class, or `Null`), and `object` no polars one.
    return None if isinstance(inner, type) or inner in (pl.Null, pl.Object) else dtype


_POLARS_FIELD = {float: pl.Float64, int: pl.Int64, bool: pl.Boolean, str: pl.String}


def _item_dtype(item: Any) -> pl.Struct:
    fields = [pl.Field(name, _POLARS_FIELD.get(base_annotation(t), pl.Object()))
              for name, t in item_schema(item)]
    return pl.Struct(fields)


_DTYPES = {t: np.dtype(d) for t, d in {float: np.float64, int: np.int64, bool: np.bool_, object: object}.items()}


def dtype_of(annotation: Any) -> np.dtype:
    """The array dtype a value of `annotation` is stored in: float64, int64, bool, else object."""
    if (dtype := _DTYPES.get(annotation)) is not None:
        return dtype
    return _DTYPES[int] if is_decimal(annotation) else _DTYPES[object]
