"""Running a tree step document in decikit over a probe frame: every feature it reads, at and around each value
it compares against, and null."""

import random

import polars as pl
from decikit import TreeConfig
from decikit.steps.trees._encode import feature_kinds
from polars.testing import assert_frame_equal
from pydantic import BaseModel

DTYPES = {"float": pl.Float64, "int": pl.Int64, "bool": pl.Boolean, "str": pl.String}


def _carriers(obj):
    """Every node or condition in `obj` that names a feature."""
    if isinstance(obj, BaseModel):
        if hasattr(obj, "feature"):
            yield obj
        for value in obj.__dict__.values():
            yield from _carriers(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _carriers(value)


def _compared(dumped, key=None):
    """The numbers and patterns a dumped condition compares against (a param's default included)."""
    if isinstance(dumped, dict):
        for k, v in dumped.items():
            if k != "feature":
                yield from _compared(v, k)
    elif isinstance(dumped, list):
        for v in dumped:
            yield from _compared(v, key)
    elif isinstance(dumped, (int, float)) and not isinstance(dumped, bool):
        yield dumped
    elif isinstance(dumped, str) and key == "patterns":
        yield dumped


def _near(value, kind: str) -> list:
    if isinstance(value, str) != (kind == "str"):
        return []
    if kind == "str":
        return [value, value.upper(), value.lower(), f" {value} ", f"x{value}", f"{value}x"]
    if kind == "int":
        return [int(value) - 1, int(value), int(value) + 1]
    return [value - 1.0, value - 0.01, float(value), value + 0.01, value + 1.0]


def probe_frame(step: dict, rows: int = 200, nulls: float = 0.1, seed: int = 0) -> pl.DataFrame:
    """`rows` rows over every feature the tree reads; row 0 is all null, later rows null a `nulls` share of values."""
    tree = TreeConfig.load(step)._tree()
    kinds = feature_kinds(tree, step.get("feature_types", {}))
    candidates = {name: {"bool": [True, False], "str": ["zz"]}.get(kind, [0]) for name, kind in kinds.items()}
    for node in tree.nodes.values():
        for carrier in _carriers(node.data):
            values = list(_compared(carrier.model_dump(mode="json", exclude={"feature"})))
            for name in carrier.feature.required_features():
                candidates[name] += [v for value in values for v in _near(value, kinds[name])]
    rng = random.Random(seed)
    columns = {name: [None] + [None if rng.random() < nulls else rng.choice(sorted(set(c), key=repr))
                               for _ in range(rows - 1)] for name, c in candidates.items()}
    # A row number keeps the frame `rows` long when the tree reads no feature at all.
    typed = {name: pl.Series(name, values, dtype=DTYPES[kinds[name]], strict=False) for name, values in columns.items()}
    return pl.DataFrame({"probe_row": range(rows), **typed})


def run(step: dict, frame: pl.DataFrame):
    """The tree's output over `frame` with its params at their defaults, or the error it raises."""
    tree = TreeConfig.load(step)
    try:
        return tree.run(frame, params=tree.parameters().defaults())
    except Exception as error:  # an export must fail the way its source does
        return f"{type(error).__name__}: {error}"


def assert_runs_alike(actual: dict, expected: dict, frame: pl.DataFrame) -> None:
    """Both step documents give the same frame (columns, order, dtypes, values), or fail the same way."""
    want, got = run(expected, frame), run(actual, frame)
    if isinstance(want, str) or isinstance(got, str):
        assert got == want
    else:
        assert_frame_equal(got, want)
