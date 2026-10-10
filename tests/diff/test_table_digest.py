"""A table default hashes in bulk: stable across processes, sensitive to every cell, and fast."""
import os
import subprocess
import sys

from decikit import ParamTable, flow, param_table
from decikit.lab import diff
from decikit.lab.diff._hash import _show, snapshot

ROWS = [{"band": i, "rate": i * 0.5, "open": i % 2 == 0} for i in range(2000)]
COLUMNS = {"band": int, "rate": float, "open": bool}
SCRIPT = "import sys; sys.path.insert(0, {here!r}); from test_table_digest import digest; print(digest())"


def digest(rows=ROWS, columns=COLUMNS) -> str:
    def rate(score: int, t: ParamTable = param_table(columns, default=rows)) -> float:
        return 0.0

    return snapshot(flow(rate, name="t")).hash


def in_process(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    return subprocess.run([sys.executable, "-c", SCRIPT.format(here=os.path.dirname(__file__))], env=env, check=True, capture_output=True, text=True).stdout.strip()


def test_the_digest_is_the_same_in_processes_with_different_hash_seeds():
    assert in_process("1") == in_process("2") == digest()


def test_changing_one_cell_changes_the_digest():
    changed = [dict(r) for r in ROWS]
    changed[1234]["rate"] += 0.25
    assert digest(changed) != digest()


def test_reordering_rows_changes_the_digest():
    assert digest(ROWS[::-1]) != digest()
    assert digest(ROWS[1:] + ROWS[:1]) != digest()


def test_swapping_two_columns_of_equal_values_changes_the_digest():
    one = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    swapped = [{"a": 2, "b": 1}, {"a": 4, "b": 3}]
    assert digest(one, {"a": int, "b": int}) != digest(swapped, {"a": int, "b": int})
    assert digest(one, {"a": int, "b": int}) != digest([{"a": 1, "b": 3}, {"a": 2, "b": 4}], {"a": int, "b": int})


def test_a_column_of_another_type_changes_the_digest():
    ints = [{"x": 1}, {"x": 2}]
    floats = [{"x": 1.0}, {"x": 2.0}]
    bools = [{"x": True}, {"x": False}]
    assert len({digest(ints, {"x": int}), digest(floats, {"x": float}), digest(bools, {"x": bool})}) == 3


def test_a_row_added_or_removed_changes_the_digest():
    assert digest(ROWS[:-1]) != digest()
    assert digest([]) != digest(ROWS[:1])


def test_a_changed_table_default_is_reported_as_a_changed_default():
    changed = [dict(r) for r in ROWS]
    changed[7]["band"] = -1

    def make(rows):
        def rate(score: int, t: ParamTable = param_table(COLUMNS, default=rows)) -> float:
            return 0.0
        return flow(rate, name="t")

    s = diff.structure(make(ROWS), make(changed))
    assert [(p.name, p.field) for p in s.params] == [("t", "default")]


def test_a_long_default_is_shown_cut_at_80_characters_without_its_whole_repr():
    assert _show(ROWS) == repr(ROWS)[:77] + "..."
    assert _show(list(range(31))) == repr(list(range(31)))[:77] + "..."
    assert _show({i: i for i in range(100)}) == repr({i: i for i in range(100)})[:77] + "..."
    assert _show([1, 2]) == "[1, 2]"


def test_a_table_default_is_hashed_per_column_not_per_cell(monkeypatch):
    from decikit.lab.diff import _hash

    calls = []
    real = _hash._feed
    monkeypatch.setattr(_hash, "_feed", lambda *a: (calls.append(1), real(*a))[1])
    digest()
    assert len(calls) < 100  # one call per cell would be 6,000 for this table
