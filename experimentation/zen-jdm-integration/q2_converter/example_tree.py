"""A small, structurally representative decider2 Tree, built directly
against the public schema classes (`decider2.trees`) — not modifying
decider2/src, only importing and instantiating it, exactly like any
consumer of the library would.

Deliberately exercises four of the six node kinds in one small tree:
CompositeNode (OR, two different features), CasesRanges (3-way numeric
band), UnaryNode (single condition, string match), LeafNode (with a
*shared* result_idx — two different leaf nodes point at the same output
row, decider2's leaf-value dedup). CasesStringMatch/CasesIsIn are exercised
separately in `q2_converter/string_match_probe.py` because they map to JDM
differently (see decider2_to_jdm.py's docstring).

    has_defaults OR income < 0            -> declined      (row 0)
    income < 5000                         -> declined      (row 0, same leaf value)
    5000 <= income < 50000, region in {GP,WC}  -> premium-high (row 2)
    5000 <= income < 50000, region not in {GP,WC} -> premium-low  (row 1)
    income >= 50000                       -> elite         (row 3)
"""
from __future__ import annotations

from decider2.trees import (
    CompositeNode,
    CasesRanges,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    RangeCondition,
    RangeEndLogic,
    TLogicOp,
    Tree,
    TreeOutput,
    UnaryGreaterThanEqual,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
)

OUTPUT_ROWS = [
    {"tier_code": 0, "limit": 0.0},  # 0: declined
    {"tier_code": 2, "limit": 75_000.0},  # 1: premium, region not in {GP,WC}
    {"tier_code": 2, "limit": 100_000.0},  # 2: premium, region in {GP,WC}
    {"tier_code": 3, "limit": 250_000.0},  # 3: elite
]


def build_tree() -> Tree:
    def edge(source: str, target: str, *idx: int) -> MultiSourceEdge:
        return MultiSourceEdge(source=source, target=target, data=MultiEdgeData(sourceIndex=list(idx)))

    nodes = [
        PositionedNode(
            id="n_gate",
            data=CompositeNode(
                id="n_gate", op=TLogicOp.OR,
                conditions=[
                    UnaryIsTrue(feature="has_defaults"),
                    UnaryLessThan(feature="income", threshold=0),
                ],
            ),
        ),
        PositionedNode(id="leaf_declined_a", data=LeafNode(id="leaf_declined_a", result_idx=0)),
        PositionedNode(
            id="n_income",
            data=CasesRanges(
                id="n_income", feature="income", end_logic=RangeEndLogic.lower_inclusive,
                conditions=[
                    RangeCondition(max=5_000),
                    RangeCondition(min=5_000, max=50_000),
                    RangeCondition(min=50_000),
                ],
            ),
        ),
        PositionedNode(id="leaf_declined_b", data=LeafNode(id="leaf_declined_b", result_idx=0)),
        # A third, structurally distinct leaf pointing at the SAME output
        # row (0) as leaf_declined_a/b — decider2 trees enforce single-
        # parent structure (found below: reusing one node via two edges
        # raises "revisits node"), so the result_idx dedup is the only
        # sharing a tree can do; the node itself is never shared.
        PositionedNode(id="leaf_declined_c", data=LeafNode(id="leaf_declined_c", result_idx=0)),
        PositionedNode(
            id="n_region",
            data=UnaryNode(id="n_region", condition=UnaryStringMatch(feature="region", patterns=["GP", "WC"])),
        ),
        PositionedNode(id="leaf_elite", data=LeafNode(id="leaf_elite", result_idx=3)),
        PositionedNode(id="leaf_premium_high", data=LeafNode(id="leaf_premium_high", result_idx=2)),
        PositionedNode(id="leaf_premium_low", data=LeafNode(id="leaf_premium_low", result_idx=1)),
    ]
    edges = [
        edge("n_gate", "leaf_declined_a", 0),
        edge("n_gate", "n_income", 1),
        edge("n_income", "leaf_declined_b", 0),
        edge("n_income", "n_region", 1),
        edge("n_income", "leaf_elite", 2),
        # CasesRanges.arity is len(conditions)+1 — one edge per band plus a
        # trailing "otherwise" (source index 3) even though the three bands
        # above are already exhaustive over the real number line. Every
        # source index a node's arity declares needs a wired edge; this one
        # is unreachable at runtime but still required at build time.
        edge("n_income", "leaf_declined_c", 3),
        edge("n_region", "leaf_premium_high", 0),
        edge("n_region", "leaf_premium_low", 1),
    ]
    output = TreeOutput(
        data=OUTPUT_ROWS,
        dtypes=[("tier_code", "Int64"), ("limit", "Float64")],
    )
    return Tree(name="credit_gate", nodes=nodes, edges=edges, output=output)


def oracle(has_defaults: bool, income: float, region: str) -> dict:
    """Independent reference implementation of the same tree, written
    directly against the spec above rather than by walking the Tree object
    or the emitted kernel — the third leg of the identical-answers check,
    same discipline as q1_benchmark/rules.py's oracle()."""
    if has_defaults or income < 0:
        row = 0
    elif income < 5_000:
        row = 0
    elif income < 50_000:
        row = 2 if region in ("GP", "WC") else 1
    else:
        row = 3
    return dict(OUTPUT_ROWS[row])
