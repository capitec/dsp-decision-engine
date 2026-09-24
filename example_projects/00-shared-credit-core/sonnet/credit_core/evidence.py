"""Decision identity and table-cell attribution (addendum A1; 09 §5.15 items 1 and 5).

Every capability that reads a table records *which version of which table,
which cell*. `decision_id` is assigned once, before any logic runs, and
carried by every downstream capability's evidence.

These are plain steps, not `ConfigurableStep`s: identity assignment and
attribution recording are arithmetic-shaped, not table-shaped.
"""
from __future__ import annotations

import uuid

from decider import param, step


def new_decision_id() -> str:
    """A globally unique id, assigned once before any capability runs (09 §5.15 item 1).

    Not a step: called by the consuming project's entry point, before the
    pipeline runs, and passed in as the `decision_id` column -- generating it
    *inside* a step would make it a second, later assignment.
    """
    return str(uuid.uuid4())


@step(output="decision_id")
def stamp_decision_id(decision_id: str = param(required=True)) -> str:
    """Carries a caller-assigned `decision_id` into the record as a column.

    Use this when `decision_id` arrives as a param (e.g. a batch run seeds
    one id per record some other way) rather than already being a frame
    column.
    """
    return decision_id


def cell_id(table: str, version: str, *keys: object) -> str:
    """A stable, content-derived cell identifier: `<table>@<version>#<key1>|<key2>|...`.

    Used as the value written to `<table>_cell_id` outputs across every
    table-backed capability, so "which cell of which table" (09 §5.15 item 5)
    is answerable the same way everywhere in the library.
    """
    key_part = "|".join(str(k) for k in keys)
    return f"{table}@{version}#{key_part}"
