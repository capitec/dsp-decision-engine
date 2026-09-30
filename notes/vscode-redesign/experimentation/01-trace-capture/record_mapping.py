"""Durable record mapping: join kernel row offsets to a `RecordRef` via driver-side keys.

The kernel writes events into a flat buffer plus per-row offsets (`offsets[r]` is
row r's first event). A kernel row index `r` is *not* a durable identity: the
driver maps `r -> RecordRef` using the input frame's declared key column(s). This
proves the join, and the two rejection rules the task asks for: duplicate keys
(no unambiguous join) and missing keys (no key to join on), plus frame-only
events (an override, a frame-step, a scatter-gather element) that have no row.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/record_mapping.py
"""
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class RecordRef:
    """A durable record identity: its declared key(s), or None for frame-scope events.

    `keys` is the tuple of key-column values (empty for a frame-scope event); the
    run's `frame_id`/`run_id` is a separate buffer constant, not per event.
    """

    keys: tuple
    row: int | None  # the kernel row index; None for a frame-scope event


def map_rows(frame: pl.DataFrame, key_cols: list[str], reject_duplicates: bool) -> tuple[dict[int, RecordRef], list[str]]:
    """`row index -> RecordRef` from the key columns; returns the map and any rejections."""
    refs: dict[int, RecordRef] = {}
    rejects: list[str] = []
    for r in range(frame.height):
        keys = tuple(frame.get_column(c)[r] for c in key_cols)
        if any(k is None for k in keys):
            rejects.append(f"row {r}: missing key in {key_cols} (null)")
            continue
        if keys in refs and reject_duplicates:
            rejects.append(f"row {r}: duplicate key {keys} already at row {refs[keys].row}")
            continue
        refs[keys] = RecordRef(keys, r)
    return {v.row: v for v in refs.values()}, rejects


if __name__ == "__main__":
    # Single key: unambiguous join of per-row offsets to RecordRef.
    frame = pl.DataFrame({"client_id": ["a", "b", "c"], "score": [1, 2, 3]})
    by_row, rejects = map_rows(frame, ["client_id"], reject_duplicates=True)
    assert rejects == []
    assert by_row[0] == RecordRef(("a",), 0) and by_row[2] == RecordRef(("c",), 2)
    print(f"single key: offsets r in [0..{frame.height - 1}] map to {sorted(v.keys for v in by_row.values())}")

    # Composite key: two columns form the identity (tenant + client_id).
    frame = pl.DataFrame({"tenant": ["t1", "t1", "t2"], "client_id": ["a", "b", "a"],
                          "score": [1, 2, 3]})
    by_row, rejects = map_rows(frame, ["tenant", "client_id"], reject_duplicates=True)
    assert rejects == [] and len(by_row) == 3
    print(f"composite key: (tenant, client_id) disambiguates the repeated client_id 'a' -> 3 distinct refs")

    # Duplicate key on a single column: rejected, not silently merged.
    frame = pl.DataFrame({"client_id": ["a", "a", "c"], "score": [1, 2, 3]})
    by_row, rejects = map_rows(frame, ["client_id"], reject_duplicates=True)
    assert rejects == ["row 1: duplicate key ('a',) already at row 0"]
    print(f"duplicate key: {rejects[0]} -> rejected (a kernel row index alone would merge two records)")

    # Missing key: rejected, and the driver falls back to a run-local row id.
    frame = pl.DataFrame({"client_id": ["a", None, "c"], "score": [1, 2, 3]})
    by_row, rejects = map_rows(frame, ["client_id"], reject_duplicates=True)
    assert any("missing key" in r for r in rejects)
    print(f"missing key: {rejects} -> driver assigns run-local row id (offsets[r] still names the row)")

    # Frame-only events carry no row: their scope is the frame/run, not a key.
    frame_only = RecordRef((), None)
    print(f"frame-only event: {frame_only} (an override or frame-step event; row is None, "
          f"the per-run frame id is a buffer constant, not a per-event field)")
