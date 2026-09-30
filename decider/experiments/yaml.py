"""A minimal YAML subset for `experiment.yaml`: nested maps, lists and scalars.

decider has no YAML dependency, and the experiment schema is data only (no
anchors, tags, flow style or multi-line scalars), so a small symmetric
loader/dumper covers it. Anything outside the subset loads through JSON
(`model_validate_json`), which is the canonical wire form.
"""
from __future__ import annotations

import json
from typing import Any


def dump(obj: Any, indent: int = 0) -> str:
    """Render a dict/list/scalar as the YAML subset `load` parses."""
    if isinstance(obj, dict):
        return "\n".join(_entries(obj, "  " * indent, indent))
    return _scalar_str(obj)


def _entries(mapping: dict, prefix: str, indent: int) -> list[str]:
    return [_entry(prefix, key, value, indent) for key, value in mapping.items()]


def _entry(prefix: str, key, value: Any, indent: int) -> str:
    key_str = _scalar_str(key)
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{key_str}: {{}}"
        return f"{prefix}{key_str}:\n" + "\n".join(_entries(value, "  " * (indent + 1), indent + 1))
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{prefix}{key_str}: []"
        return f"{prefix}{key_str}:\n" + _dump_list(value, indent + 1)
    return f"{prefix}{key_str}: {_scalar_str(value)}"


def _dump_list(items, indent: int) -> str:
    pad = "  " * indent
    lines = []
    for item in items:
        if isinstance(item, dict):
            for j, (key, value) in enumerate(item.items()):
                prefix = f"{pad}- " if j == 0 else "  " * (indent + 1)
                lines.append(_entry(prefix, key, value, indent + 1))
        else:
            lines.append(f"{pad}- {_scalar_str(item)}")
    return "\n".join(lines)


def load(text: str) -> Any:
    """Parse the YAML subset `dump` writes: nested maps, `-` lists and scalars."""
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if stripped.startswith("- "):
            lines.append((indent, True, stripped[2:].strip()))
        elif stripped == "-":
            lines.append((indent, True, ""))
        else:
            lines.append((indent, False, stripped))
    if not lines:
        return None
    value, _ = _block(lines, 0, lines[0][0])
    return value


def _block(lines, i, indent):
    if lines[i][1]:
        return _list(lines, i, indent)
    return _map(lines, i, indent)


def _map(lines, i, indent):
    out = {}
    while i < len(lines):
        ind, is_item, text = lines[i]
        if ind != indent or is_item:
            break
        key, _, val = text.partition(":")
        key = _scalar(key.strip())
        val = val.strip()
        if val == "":
            i += 1
            if i < len(lines) and lines[i][0] > indent:
                out[key], i = _block(lines, i, lines[i][0])
            else:
                out[key] = None
        else:
            out[key] = _empty(val) if val in ("[]", "{}") else _scalar(val)
            i += 1
    return out, i


def _list(lines, i, indent):
    out = []
    while i < len(lines):
        ind, is_item, text = lines[i]
        if ind != indent or not is_item:
            break
        i += 1
        if text == "":
            if i < len(lines) and lines[i][0] > indent:
                item, i = _block(lines, i, lines[i][0])
            else:
                item = None
            out.append(item)
            continue
        if ":" in text:
            key, _, val = text.partition(":")
            key = _scalar(key.strip())
            val = val.strip()
            item = {}
            if val == "":
                if i < len(lines) and lines[i][0] > indent:
                    item[key], i = _block(lines, i, lines[i][0])
                else:
                    item[key] = None
            else:
                item[key] = _empty(val) if val in ("[]", "{}") else _scalar(val)
            if i < len(lines) and lines[i][0] == indent + 2 and not lines[i][1]:
                sub, i = _map(lines, i, indent + 2)
                item.update(sub)
            out.append(item)
        else:
            out.append(_scalar(text))
    return out, i


def _empty(val: str):
    return [] if val == "[]" else {}


def _scalar(token: str) -> Any:
    t = token.strip()
    if t in ("", "null", "~"):
        return None
    if t in ("true", "True"):
        return True
    if t in ("false", "False"):
        return False
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    try:
        return float(t) if any(c in t for c in ".eE") else int(t)
    except ValueError:
        return t


def _scalar_str(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)
