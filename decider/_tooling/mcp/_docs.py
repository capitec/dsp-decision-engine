"""Config documents on disk (trees, tables, scorecards): read, check and save them for the editing tools."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from decider import ConfigurableStep


def read_doc(path: str) -> dict:
    return json.loads(Path(path).read_text())


def problems(doc: dict) -> list[str]:
    """Why `doc` would not build as a step, one line per problem; empty when it builds."""
    try:
        ConfigurableStep.load(doc).parameters()
    except ValidationError as e:
        return [f"{where}: {err['msg']}" if (where := '.'.join(map(str, err['loc']))) else err["msg"].removeprefix("Value error, ")
                for err in e.errors()]
    except Exception as e:  # noqa: BLE001  a document that fails to build is reported, not raised
        return [f"{type(e).__name__}: {e}"]
    return []


def save_doc(path: str, doc: dict) -> dict:
    """Write `doc` to `path` only if it builds; otherwise leave the file untouched and return why."""
    found = problems(doc)
    if found:
        return {"saved": False, "path": path, "problems": found}
    old = Path(path).read_text() if Path(path).exists() else "{\n  \n}\n"
    Path(path).write_text(json.dumps(doc, indent=_indent(old), ensure_ascii=False) + "\n" * old.endswith("\n"))
    return {"saved": True, "path": path}


def _indent(old: str) -> int | None:
    # Keep the file's own layout (one line or its indent width, final newline or not) so a small edit is a small diff.
    lines = old.splitlines()
    if len(lines) < 2:
        return None
    return len(lines[1]) - len(lines[1].lstrip(" ")) or 2
