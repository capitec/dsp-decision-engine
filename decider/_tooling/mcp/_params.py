"""Params documents (`configs/<version>/params.json`): view them and set values by slash path.

With an `entry` the document is checked against that pipeline's parameter schema, so an
unknown key or a missing required param is reported before the pipeline runs.
"""
from __future__ import annotations

import copy
import typing as t
from pathlib import Path

from decider.config import JsonFileStore
from decider.serving import RequestHandler
from decider._tooling.bridge.loading import load_module
from decider._tooling.mcp._configs import Refusal, json_safe, read, refusing, save


def _pipeline(path: str, entry: str) -> t.Any:
    """The flow `entry` builds for the config version holding `path`, the way `decider build` stages it.

    `entry` is `module:attr` (a flow, or a `build(...)` taking config documents by argument) or a flow file.
    """
    version = Path(path).resolve().parent
    root = version.parent.parent
    store = JsonFileStore(basepath=str(version.parent))
    try:
        config = store.read(version.name).config
    except ValueError as e:
        raise Refusal(f"{path} is not under configs/<version>/ ({e})") from None
    if entry.endswith((".yaml", ".yml", ".json")):
        return RequestHandler(store, str(root / entry)).pipeline_fn(config)
    module, _, attr = entry.partition(":")
    stem = module.replace(".", "/")
    for base in (root.parent, root):
        file = next((f for f in (base / f"{stem}.py", base / stem / "__init__.py") if f.exists()), None)
        if file:
            break
    else:
        raise Refusal(f"module '{module}' of entry '{entry}' not found under {root} or {root.parent}")
    target = load_module(str(file))
    for part in attr.split("."):
        target = getattr(target, part)
    return RequestHandler(store, target).pipeline_fn(config)


def _checked(path: str, entry: str | None, doc: dict) -> dict:
    if entry is None:
        return {}
    schema = _pipeline(path, entry).parameters()
    result = schema.validate(doc)
    return {"problems": list(result.errors), "warnings": list(result.warnings), "schema": json_safe(dict(schema))}


@refusing
def params_view(path: str, entry: str | None = None) -> dict:
    """A params document; with `entry` (e.g. "pkg.pipeline:build") also its problems against that pipeline and its schema."""
    doc = read(path)
    return {"path": path, "params": doc, **_checked(path, entry, doc)}


def _set(doc: dict, slash_path: str, value: t.Any) -> None:
    *parents, last = slash_path.strip("/").split("/")
    node: t.Any = doc
    for key in parents:
        if isinstance(node, list):
            node = node[int(key)]
        elif isinstance(node, dict):
            node = node.setdefault(key, {})
        else:
            raise Refusal(f"cannot set '{slash_path}': '{key}' is inside a {type(node).__name__}")
    if isinstance(node, list):
        node[int(last)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise Refusal(f"cannot set '{slash_path}': '{last}' is inside a {type(node).__name__}")


@refusing
def params_set(path: str, changes: dict[str, t.Any], entry: str | None = None) -> dict:
    """Set values by slash path (`{"pricing/rate/limit": 0.3}`); with `entry`, refused if it adds a problem for that pipeline."""
    old = read(path)
    doc = copy.deepcopy(old)
    try:
        for slash_path, value in changes.items():
            _set(doc, slash_path, value)
    except (ValueError, IndexError) as e:
        raise Refusal(f"bad path in changes: {e}") from None
    if entry is None:
        return save(path, doc)
    schema = _pipeline(path, entry).parameters()
    result, before = schema.validate(doc), schema.validate(old).errors
    new = [e for e in result.errors if e not in before]
    if new:
        return {"saved": False, "path": path, "problems": new}
    return {**save(path, doc), "problems": list(result.errors), "warnings": list(result.warnings)}
