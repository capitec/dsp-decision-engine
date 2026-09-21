"""Convert the sibling experiment's canonical Num/Str/Leaf tree
(`../tree-codegen-vs-interpreted/tree_shapes.py`) into a real polars
expression — a nested `pl.when(...).then(...).otherwise(...)` chain, which
is the shape doc 01 §3 calls the "balanced tree" form (not the pathological
left-leaning chain), and the shape rtlf's own README calls out as squarely
in its safe/supported subset for `.compile()`.

Reused, not duplicated: `tree_shapes.Num`/`Str`/`Leaf`/`TreeShape`/
`build_credit_tree`/`build_full_binary`/`build_one_sided_chain`/`make_rows`
are imported unmodified from the sibling directory. Nothing there is
re-implemented; this module only adds the missing third converter
(`to_schema_tree` -> decider2 codegen, `to_flat_tree` -> the interpreted
kernel, and now `to_polars_expr` -> polars).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "tree-codegen-vs-interpreted"))
from tree_shapes import Leaf, Num, Str, TreeShape  # noqa: E402

_OP_EXPR = {
    "<": lambda c, t: c < t,
    "<=": lambda c, t: c <= t,
    "==": lambda c, t: c == t,
    ">": lambda c, t: c > t,
    ">=": lambda c, t: c >= t,
    "!=": lambda c, t: c != t,
}


def _cond_expr(node) -> pl.Expr:
    if isinstance(node, Num):
        c = pl.col(node.feature)
        return _OP_EXPR[node.op](c, node.threshold)
    if isinstance(node, Str):
        return pl.col(node.feature).is_in(list(node.patterns))
    raise TypeError(node)


def _value_expr(node) -> pl.Expr:
    """Build the `then`/`otherwise` value for one node — recurses into
    nested `pl.when` chains for internal children, or a literal for a leaf."""
    if isinstance(node, Leaf):
        return pl.lit(node.value, dtype=pl.Float64)
    return pl.when(_cond_expr(node)).then(_value_expr(node.then)).otherwise(_value_expr(node.otherwise))


def to_polars_expr(shape: TreeShape) -> pl.Expr:
    """The whole tree as ONE polars expression, aliased 'score'."""
    return _value_expr(shape.root).alias("score")


def schema_for(shape: TreeShape) -> pl.Schema:
    fields = {f: pl.Float64 for f in shape.numeric_ranges}
    fields.update({f: pl.Utf8 for f in shape.string_categories})
    return pl.Schema(fields)


def rows_to_polars_df(shape: TreeShape, numeric: dict, string_codes: dict) -> pl.DataFrame:
    """Build a real polars DataFrame from the SAME row arrays
    `tree_shapes.make_rows` produced for the numba engines — numeric columns
    verbatim, string columns turned back from dictionary codes into the
    actual category strings (polars sees real `Utf8`, not decider2's int32
    dictionary-code convention, since a bare polars expression has no such
    convention — `is_in` on strings directly is the idiomatic polars form)."""
    data = {f: numeric[f] for f in sorted(shape.numeric_ranges)}
    for f in sorted(shape.string_categories):
        cats = shape.string_categories[f]
        codes = string_codes[f]
        data[f] = [cats[c] for c in codes]
    return pl.DataFrame(data, schema=schema_for(shape))
