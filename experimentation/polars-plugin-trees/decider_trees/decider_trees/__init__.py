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


_OPS = {"<": 0, "<=": 1, ">": 2, ">=": 3, "==": 4, "!=": 5, "prefix": 6, "regex": 7}


def pack(tree: list[dict]) -> tuple[bytes, list[str]]:
    """The same node list as a 36-byte-per-node blob + string table (see Rust `unpack`)."""
    import struct
    strs, out = [], []
    for n in tree:
        if "leaf" in n:
            out.append(struct.pack("<BBHIIIIqd", 0, 0, 0, 0, 0, 0, 0, n["leaf"], n["value"]))
            continue
        op = _OPS[n["op"]]
        if "f" in n:
            out.append(struct.pack("<BBHIIIIqd", 1, op, 0, n["col"], n["then"], n["else"], 0, 0, n["f"]))
        elif "i" in n:
            out.append(struct.pack("<BBHIIIIqd", 2, op, 0, n["col"], n["then"], n["else"], 0, n["i"], 0.0))
        elif "b" in n:
            out.append(struct.pack("<BBHIIIIqd", 3, op, 0, n["col"], n["then"], n["else"], 0, int(n["b"]), 0.0))
        else:
            strs.append(n["s"])
            out.append(struct.pack("<BBHIIIIqd", 4, op, 0, n["col"], n["then"], n["else"], len(strs) - 1, 0, 0.0))
    return b"".join(out), strs


def walk_value_packed(*cols, tree: list[dict], parallel: bool = False) -> pl.Expr:
    blob, strs = pack(tree)
    return register_plugin_function(plugin_path=LIB, function_name="walk_value_packed", args=list(cols),
                                    kwargs={"blob": blob, "strs": strs, "parallel": parallel}, is_elementwise=True)


def noop_packed(*cols, tree: list[dict]) -> pl.Expr:
    blob, strs = pack(tree)
    return register_plugin_function(plugin_path=LIB, function_name="noop_packed", args=list(cols),
                                    kwargs={"blob": blob, "strs": strs}, is_elementwise=True)


def noop(col) -> pl.Expr:
    return register_plugin_function(plugin_path=LIB, function_name="noop", args=[col], is_elementwise=True)


def noop_kwargs(*cols, tree, compile: bool = False) -> pl.Expr:
    """Deserialise (and optionally compile) the tree, do no work: the per-call cost of shipping it."""
    return _call("noop_compile" if compile else "noop_kwargs", cols, tree, False)


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
