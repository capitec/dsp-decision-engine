"""Independent check of the typed strand's headline correctness claim, and of
how much of it is opt-in. Run against f90488c and against the typed worktree.
"""
from pathlib import Path
import polars as pl
import decider2
from decider2 import flow
from decider2.trees import (LeafNode, MultiEdgeData, MultiSourceEdge, PositionedNode,
                            Tree, TreeOutput, UnaryEqual, UnaryNode, tree_module)

print("decider2:", Path(decider2.__file__).resolve())
BIG = 9007199254740992  # 2**53

def one_node(name):
    return Tree(name=name,
        nodes=[PositionedNode(id="root", data=UnaryNode(condition=UnaryEqual(feature="n", threshold=BIG))),
               PositionedNode(id="yes", data=LeafNode(result_idx=1)),
               PositionedNode(id="no", data=LeafNode(result_idx=0))],
        edges=[MultiSourceEdge(source="root", target="yes", data=MultiEdgeData(sourceIndex=[0])),
               MultiSourceEdge(source="root", target="no", data=MultiEdgeData(sourceIndex=[1]))],
        output=TreeOutput(data=[{"hit": 0}, {"hit": 1}], default={"hit": -1}, dtypes=[("hit", "Int64")]))

frame = pl.DataFrame({"n": pl.Series([BIG, BIG + 1], dtype=pl.Int64)})
print(f"rows [2**53, 2**53+1] tested against  n == 2**53 ;  correct answer is [1, 0]")

for label, kwargs in (("no declaration (every existing document)", {}),
                      ("declared feature_types={'n': int}", {"feature_types": {"n": int}})):
    try:
        built = tree_module(one_node("t_" + label.split()[0]), **kwargs)
    except TypeError as e:
        print(f"  {label:42s} -> API not present ({e})")
        continue
    got = {m: flow(built.module).apply(frame, mode=m)["hit"].to_list() for m in ("interpreted", "stepped", "fused")}
    ok = all(v == [1, 0] for v in got.values())
    print(f"  {label:42s} -> {got['fused']}  {'CORRECT' if ok else 'WRONG (both rows collapse in float64)'}"
          f"{'' if len(set(map(tuple, got.values()))) == 1 else '  MODES DISAGREE: ' + str(got)}")
