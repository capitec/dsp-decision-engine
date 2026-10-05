from __future__ import annotations

import dataclasses
import enum
import hashlib
import io
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from decider._engine.compile import fingerprint
from decider.contract import FlowDescription, describe
from decider.ir import CallNode, IRNode, iter_nodes, to_ir

_PARTS = ("code", "consts", "inputs", "outputs", "params")
_LITERALS = (bool, int, float, complex, str, bytes, type(None))


@dataclass(frozen=True)
class Snapshot:
    """What a diff needs of one flow, with no live objects, so it can cross a process boundary."""

    description: FlowDescription
    parts: tuple[dict[str, str], ...]  # per node, aligned with description.nodes
    decls: tuple[dict[str, dict[str, tuple[str, str]]], ...]  # per node: param -> field -> (display, digest)
    hash: str


def snapshot(step: Any) -> Snapshot:
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
    return _digest(([(k, ("array", where[v]) if isinstance(v, int) and v in where else v) for k, v in node.consts], arrays))


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
            "annotation": _pair(repr(d.annotation)),
            "default": (_show(d.default), _digest(d.default)),
            "bounds": _pair(repr(_bounds(d))),
            "shared_key": _pair(repr(d.shared_key)),
            "on_invalid": _pair(d.on_invalid),
        }
    return out


def _pair(text: str) -> tuple[str, str]:
    return text, hashlib.sha256(text.encode()).hexdigest()


def _bounds(decl: Any) -> Any:
    return getattr(decl.field_info, "metadata", None)


def _show(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _digest(value: Any) -> str:
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def _nonce() -> str:
    # A value that can't be hashed by content gets a fresh token, so it never equals another one.
    return uuid.uuid4().hex


def _function(fn: Any) -> str:
    try:
        return fingerprint(fn)
    except Exception:
        return _nonce()


def _feed(h: Any, v: Any) -> None:
    if isinstance(v, _LITERALS):
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
            _feed(h, repr(value) if f.name == "annotation" else repr(_bounds(v)) if f.name == "field_info" else value)
    elif isinstance(v, type):
        h.update(f"type:{v.__module__}.{v.__qualname__}|".encode())
    elif callable(v):
        h.update(_function(v).encode())
    else:
        h.update(_nonce().encode())
