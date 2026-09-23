"""decider_trees — a decision tree as a polars expression (Rust plugin).

    df.with_columns(walk(pl.col("income"), pl.col("sector"), tree=spec).alias("path"))

`spec` is a list of node dicts, index 0 the root:
    test(col, op, value, then, else)   -> {"col": ..., "op": ..., "f"|"i"|"b"|"s": value, "then": ..., "else": ...}
    leaf(idx, value=nan)               -> {"leaf": idx, "value": value}
`col` is the position of the feature in the `walk(...)` argument list. The
Python type of `value` picks the dtype the node tests: float -> f64, int -> i64,
bool -> bool, str -> str. A column of any other dtype is an error in Rust.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent


def test(col: int, op: str, value, then: int, else_: int) -> dict:
    if isinstance(value, bool):
        key = "b"
    elif isinstance(value, int):
        key = "i"
    elif isinstance(value, float):
        key = "f"
    elif isinstance(value, str):
        key = "s"
    else:
        raise TypeError(f"unsupported test value {value!r}")
    return {"col": col, "op": op, key: value, "then": then, "else": else_}


def leaf(idx: int, value: float = math.nan) -> dict:
    return {"leaf": idx, "value": float(value)}


def _call(name, cols, tree, parallel):
    return register_plugin_function(
        plugin_path=LIB, function_name=name, args=list(cols),
        kwargs={"nodes": tree, "parallel": parallel}, is_elementwise=True,
    )


def walk(*cols, tree: list[dict], parallel: bool = False) -> pl.Expr:
    """Leaf index reached (Int32; null where a tested feature was null)."""
    return _call("walk", cols, tree, parallel)


def walk_value(*cols, tree: list[dict], parallel: bool = False) -> pl.Expr:
    """The reached leaf's `value` (Float64)."""
    return _call("walk_value", cols, tree, parallel)


def noop(col) -> pl.Expr:
    return register_plugin_function(plugin_path=LIB, function_name="noop", args=[col], is_elementwise=True)


def panic_demo(col) -> pl.Expr:
    return register_plugin_function(plugin_path=LIB, function_name="panic_demo", args=[col], is_elementwise=True)


def to_when_then(tree: list[dict], cols: list[pl.Expr], value: bool = False) -> pl.Expr:
    """The same tree as nested pure-polars `when/then/otherwise` — the
    independent oracle the plugin is checked against (RESULTS.md §4)."""
    def cond(n):
        c = cols[n["col"]]
        op = n["op"]
        if "s" in n:
            return {"==": c == n["s"], "prefix": c.str.starts_with(n["s"]),
                    "regex": c.str.contains(n["s"])}[op]
        if "b" in n:
            return c == n["b"]
        v = n["f"] if "f" in n else n["i"]
        return {"<": c < v, "<=": c <= v, ">": c > v, ">=": c >= v, "==": c == v, "!=": c != v}[op]

    def build(k):
        n = tree[k]
        if "leaf" in n:
            return pl.lit(n["value"], dtype=pl.Float64) if value else pl.lit(n["leaf"], dtype=pl.Int32)
        c = cond(n)
        # a null feature -> null result, matching the plugin
        return pl.when(c.is_null()).then(None).when(c).then(build(n["then"])).otherwise(build(n["else"]))

    return build(0)
