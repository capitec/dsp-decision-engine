"""Random tables and records, nulls included: the compiled matcher always agrees with the Python one."""
import numpy as np
import polars as pl
import pytest

from decider.steps.tables import DecisionTableConfig

COLUMNS = {"lo": "Float64", "hi": "Float64", "key": "String", "keys": {"type": "List", "inner": "String"},
           "vals": {"type": "List", "inner": "Int64"}, "lvl": "Float64", "on": "Boolean",
           "label": "String", "pts": "Int64", "rate": "Float64"}
WORDS = ["app", "web", "branch", "", "é"]


def _expression(rng: np.random.Generator) -> dict:
    leaves = [
        {"type": "between", "variable": "score", "lower_bound_column": "lo", "upper_bound_column": "hi",
         "allow_gaps": True, "mode": str(rng.choice(["lower_inclusive", "upper_inclusive"]))},
        {"type": "eq", "variable": "channel", "value_column": "key"},
        {"type": "in", "variable": "channel", "values_column": "keys"},
        {"type": "in", "variable": "region", "values_column": "vals"},
        {"type": "eq", "variable": "level", "value_column": "lvl"},
        {"type": "eq", "variable": "flag", "value_column": "on"},
        {"type": "is_true", "variable": "flag"},
    ]
    rng.shuffle(leaves)
    while len(leaves) > 1:
        k = int(rng.integers(2, min(3, len(leaves)) + 1))
        group = {"type": str(rng.choice(["and", "or"])), "expressions": leaves[:k]}
        leaves = leaves[k:] + [group]
    return leaves[0]


def _rows(rng: np.random.Generator, n: int, nulls: bool) -> list[dict]:
    cuts = np.sort(rng.integers(0, 100, n + 1)).astype(float).tolist()

    def maybe(v):
        return None if nulls and rng.random() < 0.15 else v

    return [{"lo": None if i == 0 and rng.random() < 0.5 else cuts[i],
             "hi": None if i == n - 1 and rng.random() < 0.5 else cuts[i + 1],
             "key": maybe(str(rng.choice(WORDS))),
             "keys": maybe([str(w) for w in rng.choice(WORDS, int(rng.integers(0, 3)))]),
             "vals": maybe([int(v) for v in rng.integers(0, 5, int(rng.integers(0, 3)))]),
             "lvl": maybe(float(rng.integers(0, 3))), "on": maybe(bool(rng.random() < 0.5)),
             "label": f"row{i}", "pts": i, "rate": i / 4} for i in range(n)]


def _records(rng: np.random.Generator, n: int = 30) -> pl.DataFrame:
    def col(values, dtype):
        return pl.Series([None if rng.random() < 0.15 else v for v in values], dtype=dtype)

    return pl.DataFrame({
        "score": col(rng.integers(-5, 105, n).astype(float), pl.Float64),
        "channel": col([str(w) for w in rng.choice(WORDS + ["other"], n)], pl.String),
        "region": col(rng.integers(0, 5, n).astype(float), pl.Float64),
        "level": col(rng.integers(0, 3, n).astype(float), pl.Float64),
        "flag": col(rng.random(n) < 0.5, pl.Boolean),
    })


@pytest.mark.parametrize("seed", range(12))
def test_random_tables_agree_in_every_mode(run, seed):
    rng = np.random.default_rng(seed)
    default = None if seed % 3 == 0 else ["none", -1, 0.0]
    table = DecisionTableConfig(
        name="corpus", columns=COLUMNS, rows=_rows(rng, int(rng.integers(0, 7)), nulls=seed % 2 == 1),
        expression=_expression(rng), outputs=["label", "pts", "rate"], default=default)
    run(table, _records(rng))
