"""A tree at the size campaign analysts publish (hundreds of nodes, several conditions each) loads, compiles and traces."""
import random

import polars as pl
from decider import TreeConfig, flow

from campaign_trees.trees import problems

FEATURES = [f"f{i}" for i in range(12)]
COLUMNS = ["leaf_outcome_code", "reason_label"]


def random_tree(internal_nodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    nodes, edges = [], []

    def grow(budget: int, depth: int) -> str:
        nid = f"x{len(nodes)}"
        if budget <= 0 or depth >= 12:
            nodes.append({"id": nid, "data": {"type": "leaf", "result_idx": rng.randrange(8)}})
            return nid
        conditions = [{"op": rng.choice([">=", "<="]), "feature": rng.choice(FEATURES), "threshold": round(rng.random(), 2)}
                      for _ in range(rng.randint(1, 4))]
        nodes.append({"id": nid, "data": {"type": "composite", "op": rng.choice(["and", "and", "or"]), "conditions": conditions}})
        left = grow(budget // 2 - 1, depth + 1)
        right = grow(budget - budget // 2, depth + 1)
        edges.extend({"source": nid, "target": t, "data": {"sourceIndex": i}} for i, t in enumerate((left, right)))
        return nid

    grow(internal_nodes, 1)
    rows = [{"leaf_outcome_code": i % 2, "reason_label": 9100 + i} for i in range(8)]
    return {"type": "tree", "name": "big", "path_output": "leaf", "trace_output": "path", "tree": {
        "nodes": nodes, "edges": edges,
        "output": {"data": rows, "default": rows[0], "dtypes": [["leaf_outcome_code", "Int64"], ["reason_label", "Int64"]]}}}


def test_a_300_node_tree_is_complete_and_every_client_ends_in_a_leaf():
    tree = TreeConfig.load(random_tree(150, seed=7))
    assert len(tree.nodes) > 250 and problems(tree) == []
    rng = random.Random(1)
    frame = pl.DataFrame({f: [rng.random() for _ in range(500)] for f in FEATURES})
    paths = tree.run(frame)["path"].to_list()
    assert all(path.split(">")[-1].startswith("x") and tree.nodes[path.split(">")[-1]].children == () for path in paths)
    assert all(path.split(">")[0] == "x0" for path in paths)


def test_the_compiled_tree_gives_the_interpreted_paths():
    tree = TreeConfig.load(random_tree(150, seed=7))
    rng = random.Random(2)
    frame = pl.DataFrame({f: [rng.random() for _ in range(200)] for f in FEATURES})
    pipeline = flow(tree, name="big")
    assert pipeline.compile("fused").run(frame)["path"].to_list() == pipeline.compile("interpreted").run(frame)["path"].to_list()
