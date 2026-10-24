"""Decision identity and table-cell attribution shared by every table-backed capability."""
from __future__ import annotations

import uuid


def new_decision_id() -> str:
    """A globally unique id. Assign it once, before the pipeline runs, and pass it in as `decision_id`.

    Example: `{"decision_id": new_decision_id(), ...}`
    """
    return str(uuid.uuid4())


def cell_id(table: str, version: str, *keys: object) -> str:
    """The stable id of one table cell: `cell_id("tax", "v1", 2) == "tax@v1#2"`."""
    return f"{table}@{version}#{'|'.join(map(str, keys))}"
