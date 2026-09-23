"""The 4-step pipeline (tree -> arithmetic -> tree -> threshold), written ONCE
as plugin-style specs and rendered three ways: the polars plugin, pure polars
`when/then` (oracle), and decider2's own Tree documents + steps."""
from __future__ import annotations

import common  # noqa: F401  (puts the built decider_trees package on sys.path)
import numpy as np
import polars as pl

import decider_trees as dt

# T1 features, in argument order: 0 income f64, 1 age i64, 2 sector str, 3 verified bool
T1 = [
    dt.test(0, "<", 5000.0, 1, 2),      # 0
    dt.test(1, "<", 25, 8, 3),          # 1
    dt.test(2, "==", "mining", 4, 5),   # 2
    dt.test(3, "==", True, 9, 10),      # 3
    dt.test(1, ">=", 40, 11, 12),       # 4
    dt.test(0, ">=", 20000.0, 13, 6),   # 5
    dt.test(2, "==", "retail", 14, 7),  # 6
    dt.test(1, "<", 60, 15, 16),        # 7
    dt.leaf(0, 5.0), dt.leaf(1, 10.0), dt.leaf(2, 8.0), dt.leaf(3, 20.0),   # 8..11
    dt.leaf(4, 15.0), dt.leaf(5, 30.0), dt.leaf(6, 12.0), dt.leaf(7, 18.0), dt.leaf(8, 14.0),  # 12..16
]
T1_FEATURES = ["income", "age", "sector", "verified"]

# T2 features: 0 adj f64, 1 sector str
T2 = [
    dt.test(0, "<", 100.0, 5, 1),         # 0
    dt.test(0, "<", 300.0, 2, 3),         # 1
    dt.test(1, "==", "mining", 6, 7),     # 2
    dt.test(0, ">=", 1000.0, 8, 9),       # 3
    dt.leaf(0, 5.0),                      # 4 (unused leaf keeps idx dense) -> not referenced
    dt.leaf(0, 5.0), dt.leaf(1, 12.0), dt.leaf(2, 16.0), dt.leaf(3, 25.0), dt.leaf(4, 20.0),  # 5..9
]
T2_FEATURES = ["adj", "sector"]
CUTOFF = 15.0
SECTORS = ["retail", "mining", "finance", "agri", "retail-online", "public"]


def make_frame(n: int, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "income": np.round(rng.lognormal(9.0, 0.8, n), 2),                       # f64
        "age": rng.integers(18, 76, n, dtype=np.int64),                           # i64
        "sector": pl.Series(rng.choice(SECTORS, n).tolist(), dtype=pl.String),   # str
        "verified": rng.random(n) < 0.7,                                          # bool
    })


# ---------------------------------------------------------------- polars side
def polars_exprs(parallel: bool = False, oracle: bool = False):
    """The four expressions. Each depends on the previous one's column, so they
    are four `with_columns` passes (four intermediates), not one."""
    c = pl.col
    if oracle:
        s1 = dt.to_when_then(T1, [c(f) for f in T1_FEATURES], value=True)
        s3 = dt.to_when_then(T2, [c(f) for f in T2_FEATURES], value=True)
    else:
        s1 = dt.walk_value(*[c(f) for f in T1_FEATURES], tree=T1, parallel=parallel)
        s3 = dt.walk_value(*[c(f) for f in T2_FEATURES], tree=T2, parallel=parallel)
    return [
        s1.alias("pts1"),
        (c("pts1") * c("income") / 1000.0 + c("age")).alias("adj"),
        s3.alias("pts2"),
        (c("pts2") >= CUTOFF).alias("approve"),
    ]


def run_polars(df: pl.DataFrame, how: str = "eager", parallel: bool = False, oracle: bool = False) -> pl.DataFrame:
    e = polars_exprs(parallel, oracle)
    if how == "eager":
        for x in e:
            df = df.with_columns(x)
        return df
    lf = df.lazy()
    for x in e:
        lf = lf.with_columns(x)
    if how == "lazy":
        return lf.collect()
    if how == "streaming":
        return lf.collect(engine="streaming")
    raise ValueError(how)


# --------------------------------------------------------------- decider2 side
def to_decider2_tree(spec, features, name, out_col):
    from decider2.trees import (LeafNode, MultiEdgeData, MultiSourceEdge, PositionedNode, Tree,
                                TreeOutput, UnaryNode, UnaryLessThan, UnaryLessThanEqual, UnaryGreaterThan,
                                UnaryGreaterThanEqual, UnaryEqual, UnaryNotEqual, UnaryIsTrue, UnaryIsFalse,
                                UnaryStringMatch)
    num = {"<": UnaryLessThan, "<=": UnaryLessThanEqual, ">": UnaryGreaterThan, ">=": UnaryGreaterThanEqual,
           "==": UnaryEqual, "!=": UnaryNotEqual}
    nodes, edges, leaves = [], [], {}
    for k, n in enumerate(spec):
        nid = f"n{k}"
        if "leaf" in n:
            nodes.append(PositionedNode(id=nid, data=LeafNode(result_idx=n["leaf"])))
            leaves[n["leaf"]] = n["value"]
            continue
        feat = features[n["col"]]
        if "f" in n or "i" in n:
            v = n["f"] if "f" in n else n["i"]
            cond = num[n["op"]](feature=feat, threshold=float(v))
        elif "b" in n:
            cond = (UnaryIsTrue if n["b"] else UnaryIsFalse)(feature=feat)
        else:
            assert n["op"] == "==", "decider2 supports exact string match only"
            cond = UnaryStringMatch(feature=feat, patterns=[n["s"]])
        nodes.append(PositionedNode(id=nid, data=UnaryNode(condition=cond)))
        edges.append(MultiSourceEdge(source=nid, target=f"n{n['then']}", data=MultiEdgeData(sourceIndex=[0])))
        edges.append(MultiSourceEdge(source=nid, target=f"n{n['else']}", data=MultiEdgeData(sourceIndex=[1])))
    data = [{out_col: leaves[i]} for i in range(max(leaves) + 1)]
    return Tree(name=name, nodes=nodes, edges=edges,
                output=TreeOutput(data=data, default={out_col: float("nan")}, dtypes=[(out_col, "Float64")]))


def decider2_pipeline():
    from decider2 import flow, param
    from decider2.trees import tree_module

    t1 = tree_module(to_decider2_tree(T1, T1_FEATURES, "band", "pts1"))
    t2 = tree_module(to_decider2_tree(T2, T2_FEATURES, "grade", "pts2"))

    def adj(pts1: float, income: float, age: int) -> float:
        """Points scaled by income, plus age."""
        return pts1 * income / 1000.0 + age

    def approve(pts2: float, cutoff: float = param(CUTOFF)) -> bool:
        """Approve above the cutoff."""
        return pts2 >= cutoff

    return flow(t1.module, adj, t2.module, approve).emit("pts1", "adj", "pts2", "band_path", "grade_path")
