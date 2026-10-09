from __future__ import annotations

import dataclasses
import enum
import hashlib
import io
import uuid
from dataclasses import dataclass
from itertools import islice
from operator import itemgetter
from typing import Any

import numpy as np
import polars as pl

from decider._engine.compile import fingerprint
from decider.lab._refs import FlowDescription
from decider.lab._describe import describe
from decider.ir import CallNode, IRNode, iter_nodes, to_ir

_PARTS = ("code", "consts", "inputs", "outputs", "params")
_DTYPES = {"int": np.int64, "float": np.float64, "bool": np.bool_}
_ROWS: dict[Any, tuple[list, bytes]] = {}
_LITERALS = (bool, int, float, complex, str, bytes, type(None))


@dataclass(frozen=True)
class Snapshot:
    """What a diff needs of one flow, with no live objects, so it can cross a process boundary."""

    description: FlowDescription
    parts: tuple[dict[str, str], ...]  # per node, aligned with description.nodes
    decls: tuple[dict[str, dict[str, tuple[str, str]]], ...]  # per node: param -> field -> (display, digest)
    hash: str


def snapshot(step: Any) -> Snapshot:
    try:
        return _snapshot(step)
    finally:
        _ROWS.clear()  # a default edited between snapshots must hash again


def _snapshot(step: Any) -> Snapshot:
    ir = to_ir(step)
    nodes = list(iter_nodes(ir))
    parts = tuple(_parts(n) for n in nodes)
    whole = hashlib.sha256()
    for node, p in zip(nodes, parts):
        # The shape: node type and child count fix the tree, since nodes come in pre-order.
        whole.update(f"{type(node).__name__}/{len(node.children())}".encode())
        for name in _PARTS:
            whole.update(p.get(name, "").encode())
        if "other" in p:
            whole.update(p["other"].encode())
    return Snapshot(describe(step), parts, tuple(_decls(n) for n in nodes), whole.hexdigest())


def _parts(node: IRNode) -> dict[str, str]:
    if not isinstance(node, CallNode):
        return {"other": _digest(_fields(node))}
    # A reference that carries packed arrays is the Python twin of the kernel those arrays drive, and
    # closes over objects fingerprint can only key by identity, so the arrays stand for it.
    # ponytail: trusts the twin to be built from the same arrays; drop the shortcut if one ever isn't.
    by_content = node.reference is not None and not getattr(node.reference, "arrays", None)
    code = (node.kind, node.nogil, _function(node.fn), _function(node.reference) if by_content else None)
    return {"code": _digest(code), "consts": _consts(node), "inputs": _digest(node.inputs),
            "outputs": _digest(node.outputs), "params": _digest(node.params)}


def _consts(node: CallNode) -> str:
    # Trees and tables pass their packed arrays as raw addresses, which differ per build; the
    # reference holds the arrays, so hash their content and name each address by its array's position.
    arrays = getattr(node.reference, "arrays", [])
    where = {a.ctypes.data: i for i, a in enumerate(arrays)}

    def named(v: Any) -> Any:
        if isinstance(v, int):
            return ("array", where[v]) if v in where else v
        return tuple(map(named, v)) if isinstance(v, tuple) else v  # a packed table is a tuple of addresses

    return _digest(([(k, named(v)) for k, v in node.consts], arrays))


def _fields(node: IRNode) -> list[Any]:
    # Child nodes are the tree shape, hashed by position; what is left is the node's own settings.
    own = []
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        if f.name == "origin" or isinstance(value, IRNode) or (isinstance(value, tuple) and any(isinstance(v, IRNode) for v in value)):
            continue
        own.append((f.name, value))
    return own


def _decls(node: IRNode) -> dict[str, dict[str, tuple[str, str]]]:
    out = {}
    for d in getattr(node, "params", ()):
        out[d.name] = {
            "annotation": _pair(_annotation(d)),
            "default": (_show(d.default), _digest(d.default, d.schema)),
            "bounds": _pair(repr(_bounds(d))),
            "shared_key": _pair(repr(d.shared_key)),
            "on_invalid": _pair(d.on_invalid),
        }
    return out


def _pair(text: str) -> tuple[str, str]:
    return text, hashlib.sha256(text.encode()).hexdigest()


def _annotation(decl: Any) -> str:
    # A table's annotation is built from its schema and its repr holds a function address, which differs per process.
    schema = getattr(decl, "schema", None)
    return repr(decl.annotation) if schema is None else f"table{schema}"


def _bounds(decl: Any) -> Any:
    return getattr(decl.field_info, "metadata", None)


def _show(value: Any) -> str:
    # Thirty items already fill 80 characters, so a long table default never needs its whole repr.
    cut = isinstance(value, (list, tuple, dict)) and len(value) > 30
    if cut:
        value = dict(islice(value.items(), 30)) if isinstance(value, dict) else value[:30]
    text = repr(value)
    return text if len(text) <= 80 and not cut else text[:77] + "..."


def _digest(value: Any, schema: Any = None) -> str:
    h = hashlib.sha256()
    _feed(h, value, schema)
    return h.hexdigest()


def _feed_rows(h: Any, rows: list, schema: Any) -> None:
    # A table's rows hash as one buffer per column: a Python call per cell costs seconds on a 60,000-row table.
    # Nodes share one default list, so a snapshot hashes it once; it keeps `rows` alive so its id stays unique.
    key = id(rows), schema
    if key not in _ROWS:
        part = hashlib.sha256(f"rows[{len(rows)}]".encode())
        for column, kind in schema:
            values = np.fromiter(map(itemgetter(column), rows), _DTYPES[kind], len(rows))
            part.update(f"{column}:{values.dtype.str}|".encode())
            part.update(values.data)
        _ROWS[key] = rows, part.digest()
    h.update(_ROWS[key][1])


def _nonce() -> str:
    # A value that can't be hashed by content gets a fresh token, so it never equals another one.
    return uuid.uuid4().hex


def _function(fn: Any) -> str:
    try:
        return fingerprint(fn, strict=True)
    except Exception:
        return _nonce()


def _feed(h: Any, v: Any, schema: Any = None) -> None:
    if schema and isinstance(v, list) and all(kind in _DTYPES for _, kind in schema):
        _feed_rows(h, v, schema)
    elif isinstance(v, _LITERALS):
        h.update(f"{type(v).__name__}:{v!r}|".encode())
    elif isinstance(v, (tuple, list)):
        h.update(f"{type(v).__name__}[{len(v)}]".encode())
        for item in v:
            _feed(h, item)
    elif isinstance(v, dict):
        h.update(f"dict[{len(v)}]".encode())
        for k in sorted(v, key=repr):
            _feed(h, k)
            _feed(h, v[k])
    elif isinstance(v, enum.Enum):
        h.update(f"enum:{type(v).__qualname__}.{v.name}|".encode())
    elif isinstance(v, np.ndarray) and v.dtype != object:
        h.update(f"ndarray:{v.dtype.str}:{v.shape}|".encode())
        h.update(np.ascontiguousarray(v).data)
    elif isinstance(v, pl.DataFrame):
        buf = io.BytesIO()
        v.write_ipc(buf)
        h.update(buf.getvalue())
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        h.update(f"dataclass:{type(v).__qualname__}|".encode())
        for f in dataclasses.fields(v):
            value = getattr(v, f.name)
            # An annotation hashes by its spelling and a FieldInfo by its bounds: neither has a content hash.
            _feed(h, _annotation(v) if f.name == "annotation" else repr(_bounds(v)) if f.name == "field_info" else value,
                  getattr(v, "schema", None) if f.name == "default" else None)
    elif isinstance(v, type):
        h.update(f"type:{v.__module__}.{v.__qualname__}|".encode())
    elif callable(v):
        h.update(_function(v).encode())
    else:
        h.update(_nonce().encode())
