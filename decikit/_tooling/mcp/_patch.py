"""RFC 6902 JSON Patch over parsed JSON, in plain Python."""
from __future__ import annotations

import copy
import typing as t


class PatchError(ValueError):
    """A patch op that cannot apply; `index` is its place in the op list."""

    def __init__(self, index: int, op: t.Any, message: str):
        pointer = op.get("path") if isinstance(op, dict) else None
        super().__init__(f"op {index} ({op.get('op') if isinstance(op, dict) else op!r} at {pointer!r}): {message}")


def apply_patch(doc: t.Any, ops: list[dict]) -> t.Any:
    """A copy of `doc` with `ops` applied in order; raises `PatchError` naming the first op that fails."""
    doc = copy.deepcopy(doc)
    for i, op in enumerate(ops):
        try:
            doc = _apply(doc, op)
        except (ValueError, KeyError, TypeError) as e:
            raise PatchError(i, op, e.args[0] if e.args else repr(e)) from None
    return doc


def _apply(doc: t.Any, op: dict) -> t.Any:
    kind, path = op["op"], _tokens(op["path"])
    if kind in ("move", "copy"):
        source = _tokens(op["from"])
        value = copy.deepcopy(_get(doc, source))
        if kind == "move":
            if path[: len(source)] == source and len(path) > len(source):
                raise ValueError("cannot move a value into itself")
            doc = _remove(doc, source)
        return _add(doc, path, value)
    if kind == "add":
        return _add(doc, path, op["value"])
    if kind == "remove":
        return _remove(doc, path)
    if kind == "replace":
        _get(doc, path)
        return _set(doc, path, op["value"])
    if kind == "test":
        found = _get(doc, path)
        if found != op["value"]:
            raise ValueError(f"test failed: the value there is {found!r}")
        return doc
    raise ValueError(f"unknown op {kind!r}; use add, remove, replace, move, copy or test")


def _tokens(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("a JSON Pointer is empty or starts with '/'")
    return [s.replace("~1", "/").replace("~0", "~") for s in pointer[1:].split("/")]


def _index(items: list, token: str, end: bool = False) -> int:
    if end and token == "-":
        return len(items)
    if not token.isdigit():
        raise ValueError(f"{token!r} is not an array index")
    i = int(token)
    if i >= len(items) + end:
        raise ValueError(f"index {i} is out of range for an array of {len(items)}")
    return i


def _get(doc: t.Any, tokens: list[str]) -> t.Any:
    for tok in tokens:
        if isinstance(doc, dict):
            if tok not in doc:
                raise ValueError(f"no key {tok!r}; keys are {list(doc)[:12]}")
            doc = doc[tok]
        elif isinstance(doc, list):
            doc = doc[_index(doc, tok)]
        else:
            raise ValueError(f"cannot look up {tok!r} in a {type(doc).__name__}")
    return doc


def _set(doc: t.Any, tokens: list[str], value: t.Any) -> t.Any:
    if not tokens:
        return value
    parent = _get(doc, tokens[:-1])
    if isinstance(parent, dict):
        parent[tokens[-1]] = value
    else:
        parent[_index(parent, tokens[-1])] = value
    return doc


def _add(doc: t.Any, tokens: list[str], value: t.Any) -> t.Any:
    if not tokens:
        return value
    parent = _get(doc, tokens[:-1])
    if isinstance(parent, dict):
        parent[tokens[-1]] = value
    elif isinstance(parent, list):
        parent.insert(_index(parent, tokens[-1], end=True), value)
    else:
        raise ValueError(f"cannot add into a {type(parent).__name__}")
    return doc


def _remove(doc: t.Any, tokens: list[str]) -> t.Any:
    if not tokens:
        raise ValueError("cannot remove the whole document")
    _get(doc, tokens)
    parent = _get(doc, tokens[:-1])
    if isinstance(parent, dict):
        del parent[tokens[-1]]
    else:
        del parent[_index(parent, tokens[-1])]
    return doc
