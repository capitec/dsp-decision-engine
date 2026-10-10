"""Config documents as files: list, read, validate, write, patch and try them.

Step documents (trees, tables, scorecards, custom types) are checked by building
them; a `params.json` is a plain JSON object, checked against a pipeline only
when the caller names one (see `_params`).
"""
from __future__ import annotations

import datetime
import decimal
import functools
import json
import typing as t
from pathlib import Path

import polars as pl

from decikit import ConfigurableStep
from decikit._tooling.mcp import _docs
from decikit._tooling.mcp._patch import PatchError, apply_patch
from decikit._tooling.mcp._tools import _EXCLUDED


class Refusal(ValueError):
    """An edit the tools decline to make; the message says why and what to do instead."""


def refusing(fn: t.Callable[..., dict]) -> t.Callable[..., dict]:
    """Turn a `Refusal` raised by the editing tool `fn(path, ...)` into the usual unsaved reply with `problems`."""
    @functools.wraps(fn)
    def wrapper(path: str, *args: t.Any, **kwargs: t.Any) -> dict:
        try:
            return fn(path, *args, **kwargs)
        except Refusal as e:
            return {"saved": False, "path": path, "problems": [str(e)]}
    return wrapper


def is_params(path: str) -> bool:
    return Path(path).name == "params.json"


def read(path: str) -> t.Any:
    return _docs.read_doc(path)


def check(path: str, doc: t.Any) -> list[str]:
    """Why `doc` is not a valid document for `path`; empty when it is."""
    if is_params(path):
        return [] if isinstance(doc, dict) else ["a params document is a JSON object keyed by step path"]
    if not isinstance(doc, dict):
        return ["a config document is a JSON object with a top-level 'type'"]
    return _docs.problems(doc)


def save(path: str, doc: t.Any) -> dict:
    """Write `doc` to `path` if `check` passes (keeping the file's layout); otherwise leave the file alone."""
    if not is_params(path):
        return _docs.save_doc(path, doc) if isinstance(doc, dict) else \
            {"saved": False, "path": path, "problems": check(path, doc)}
    found = check(path, doc)
    if found:
        return {"saved": False, "path": path, "problems": found}
    old = Path(path).read_text() if Path(path).exists() else "{\n  \n}\n"
    Path(path).write_text(json.dumps(doc, indent=_docs._indent(old), ensure_ascii=False) + "\n" * old.endswith("\n"))
    return {"saved": True, "path": path}


def list_configs(root: str = ".") -> dict:
    """Every config document under `root`: path, kind (the step `type`, or `params`), name, version and key."""
    base = Path(root).resolve()
    found = []
    for path in sorted(base.rglob("*.json")):
        rel = path.relative_to(base).parts
        if any(p in _EXCLUDED or p.startswith(".") for p in rel[:-1]):
            continue
        entry = _entry(path)
        if entry:
            found.append(entry)
    return {"configs": found}


def _entry(path: Path) -> dict | None:
    version = key = None
    parts = path.parts
    under = ()
    if "configs" in parts[:-2]:
        i = len(parts) - 1 - parts[::-1].index("configs")
        version, under = parts[i + 1], parts[i + 2:]
        key = ".".join(under)[: -len(".json")]
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if version and isinstance(doc, dict) and under == ("params.json",):
        return {"path": str(path), "kind": "params", "name": "params", "version": version, "key": key}
    tag = doc.get("type") if isinstance(doc, dict) else None
    if not isinstance(tag, str):
        return None
    try:
        ConfigurableStep.resolve(tag)
    except Exception:  # noqa: BLE001  not a registered step type: some other JSON file
        return None
    return {"path": str(path), "kind": tag, "name": doc.get("name"), "version": version, "key": key}


def read_config(path: str) -> dict:
    """The document at `path` and `problems` (why it would not build; empty when it does)."""
    doc = read(path)
    return {"path": path, "doc": doc, "problems": check(path, doc)}


def validate_config(path: str | None = None, doc: dict | None = None) -> dict:
    """`problems` for the file at `path`, or for an unsaved `doc` (give one of them)."""
    if (path is None) == (doc is None):
        raise ValueError("give exactly one of path or doc")
    found = check(path or "", read(path) if doc is None else doc)
    return {"valid": not found, "problems": found}


def write_config(path: str, doc: dict, overwrite: bool = False) -> dict:
    """Create the document at `path` (parent directories included); an existing file needs `overwrite`."""
    if Path(path).exists() and not overwrite:
        return {"saved": False, "path": path, "problems": [f"{path} exists; pass overwrite=True to replace it"]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return save(path, doc)


def patch_config(path: str, ops: list[dict], dry_run: bool = False) -> dict:
    """Apply RFC 6902 `ops` to the document; refuse (file untouched) if an op fails or the result would not build."""
    try:
        patched = apply_patch(read(path), ops)
    except PatchError as e:
        return {"saved": False, "path": path, "problems": [str(e)]}
    if dry_run:
        return {"saved": False, "dry_run": True, "path": path, "doc": patched, "problems": check(path, patched)}
    return save(path, patched)


def try_config(path: str, rows: list[dict], params: dict | None = None) -> dict:
    """Run the document's step interpreted on `rows` and return the output rows (inputs and outputs), or `problems`."""
    doc = read(path)
    found = check(path, doc)
    if found:
        return {"problems": found}
    out = ConfigurableStep.load(doc).run(pl.DataFrame(rows), params=params)
    return {"columns": out.columns, "rows": json_safe(out.to_dicts())}


def _default(x: t.Any) -> t.Any:
    if isinstance(x, (datetime.date, datetime.datetime, datetime.time)):
        return x.isoformat()
    return str(x) if isinstance(x, decimal.Decimal) else repr(x)


def json_safe(x: t.Any) -> t.Any:
    """`x` as plain JSON values: dates as ISO strings, decimals as digit strings."""
    return json.loads(json.dumps(x, default=_default))
