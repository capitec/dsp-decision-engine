"""Experiment 02 measurements: node/edge counts, nesting depth and payload size.

Measures the real retail-credit fixture and generated dense IR at target sizes.
The edge-count rules mirror `tools/decider-ui/src/layout.ts` (`orderEdges`,
`dataEdges`) so the numbers match what the webview actually draws.

Run: uv run python notes/vscode-redesign/experimentation/02-flow-scale/measure.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

from decider.config import JsonFileStore
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import BranchNode, CallNode, IRNode, LoopNode, SequenceNode
from decider.engine.ir.origin import Origin
from decider.serving.handler import RequestHandler

ROOT = Path(__file__).resolve().parents[4]  # the worktree root


# ---- edge-count rules ported from layout.ts ---------------------------------

def arm_keys(ir: IRNode) -> dict[str, dict[str, int]]:
    keys: dict[str, dict[str, int]] = {}
    def visit(n: IRNode, inherited: dict[str, int]) -> None:
        keys[n.origin.path] = inherited
        if isinstance(n, CallNode):
            return
        for i, c in enumerate(n.children()):
            own = {**inherited, n.origin.path: i} if isinstance(n, BranchNode) and i > 0 else inherited
            visit(c, own)
    visit(ir, {})
    return keys


def exclusive(a: dict[str, int], b: dict[str, int]) -> bool:
    for branch, arm in a.items():
        if branch in b and b[branch] != arm:
            return True
    return False


def call_nodes(ir: IRNode) -> list[CallNode]:
    out: list[CallNode] = []
    def visit(n: IRNode) -> None:
        if isinstance(n, CallNode):
            out.append(n)
        else:
            for c in n.children():
                visit(c)
    visit(ir)
    return out


def data_edges(ir: IRNode) -> int:
    arms = arm_keys(ir)
    seen: list[CallNode] = []
    count = 0
    for node in call_nodes(ir):
        for inp in node.inputs or ():
            for w in reversed(seen):
                if inp.name in (o.name for o in (w.outputs or ())) and not exclusive(arms[w.origin.path], arms[node.origin.path]):
                    count += 1
                    break
        seen.append(node)
    return count


def is_leaf(n: IRNode) -> bool:
    return isinstance(n, CallNode)


def first_leaf(n: IRNode) -> str:
    return n.origin.path if is_leaf(n) else first_leaf(n.children()[0])


def last_leaves(n: IRNode) -> list[str]:
    if is_leaf(n):
        return [n.origin.path]
    if isinstance(n, BranchNode):
        return [p for arm in n.arms for p in last_leaves(arm)]
    if isinstance(n, LoopNode):
        return [n.condition.origin.path]
    return last_leaves(n.children()[-1])


def order_edges(ir: IRNode) -> int:
    count = 0
    def walk(n: IRNode) -> None:
        nonlocal count
        if is_leaf(n):
            return
        if isinstance(n, SequenceNode):
            for i in range(1, len(n.children())):
                count += len(last_leaves(n.children()[i - 1]))
        elif isinstance(n, BranchNode):
            count += len(n.arms)
        elif isinstance(n, LoopNode):
            count += 1 + len(last_leaves(n.body))  # cond->body plus body->cond back edges
        for c in n.children():
            walk(c)
    walk(ir)
    return count


def stats(ir: IRNode) -> dict:
    calls = call_nodes(ir)
    groups = []
    depth = [0]
    def walk(n: IRNode, d: int) -> None:
        depth[0] = max(depth[0], d)
        if not isinstance(n, CallNode):
            groups.append(n)
            for c in n.children():
                walk(c, d + 1)
    walk(ir, 0)
    return {
        "calls": len(calls),
        "groups": len(groups),
        "total": len(calls) + len(groups),
        "max_depth": depth[0],
        "inputs": sum(len(c.inputs or ()) for c in calls),
        "outputs": sum(len(c.outputs or ()) for c in calls),
        "order_edges": order_edges(ir),
        "data_edges": data_edges(ir),
        "group_kinds": dict(Counter(type(n).__name__ for n in groups)),
    }


def payload_bytes(ir: IRNode) -> int:
    def node(n: IRNode) -> dict:
        base = {"path": n.origin.path, "source": n.origin.source, "file": None, "line": None}
        if isinstance(n, CallNode):
            return {**base, "kind": "call",
                    "inputs": None if n.inputs is None else [i.name for i in n.inputs],
                    "outputs": None if n.outputs is None else [o.name for o in n.outputs],
                    "params": {p.name: p.default for p in n.params}}
        kind = "branch" if isinstance(n, BranchNode) else "loop" if isinstance(n, LoopNode) else "sequence"
        return {**base, "kind": kind, "children": [node(c) for c in n.children()]}
    return len(json.dumps(node(ir)))


# ---- the real fixture --------------------------------------------------------

def real_fixture():
    sys.path.insert(0, str((ROOT / "example_projects/00-shared-credit-core/sonnet").resolve()))
    sys.path.insert(0, str((ROOT / "example_projects/10-retail-credit-e2e/sonnet").resolve()))
    import pipeline  # noqa: F401
    proj = (ROOT / "example_projects/10-retail-credit-e2e/sonnet").resolve()
    store = JsonFileStore(basepath=str(proj / "configs"))
    version = store.latest_version()
    config = store.read(version).config if version else {}
    step = RequestHandler(JsonFileStore(), pipeline.build).pipeline_fn(config)
    return to_ir(step)


# ---- generated dense IR ------------------------------------------------------

def _origin(path: str) -> Origin:
    return Origin(path=path, source=path)


def _call(path: str, n_cols: int) -> CallNode:
    from decider.engine.ir.decls import Input, Output
    cols = tuple(Output(f"c{i}", str) for i in range(n_cols))
    inputs = tuple(Input(f"c{i}", str) for i in range(min(4, n_cols)))
    return CallNode(_origin(path), "scalar", (lambda: None), inputs, cols, ())


def gen_dense(leaf_nodes: int, depth: int, fanout: int, n_cols: int) -> IRNode:
    """A nested tree of sequences with exactly `depth` group levels holding
    `leaf_nodes` call nodes in total. Every call writes `n_cols` shared columns
    and reads the first four, so data edges stay dense (each call feeds the next)."""
    calls: list[CallNode] = [_call(f"n/{i}", n_cols) for i in range(leaf_nodes)]

    def build(calls_: list[CallNode], remaining_depth: int, path: str) -> IRNode:
        if remaining_depth <= 0 or len(calls_) <= 1:
            return SequenceNode(_origin(path), tuple(calls_))
        kids = [build(calls_[i::fanout], remaining_depth - 1, f"{path}/{i}") for i in range(fanout) if calls_[i::fanout]]
        return SequenceNode(_origin(path), tuple(kids))

    return build(calls, depth, "root")


def main() -> None:
    real = real_fixture()
    r = stats(real)
    print("== real fixture (10-retail-credit-e2e) ==")
    print(json.dumps(r, indent=2))
    print("ir payload bytes:", payload_bytes(real))

    print("\n== generated dense IR ==")
    for label, leaf, depth, fanout, ncols in [
        ("1k nodes / depth 6", 1000, 6, 4, 8),
        ("2k nodes / depth 8", 2000, 8, 3, 8),
        ("5k nodes / depth 8", 5000, 8, 3, 8),
        ("10k nodes / depth 10", 10000, 10, 3, 8),
    ]:
        ir = gen_dense(leaf, depth, fanout, ncols)
        t0 = time.perf_counter()
        s = stats(ir)
        s["payload_bytes"] = payload_bytes(ir)
        s["stats_sec"] = round(time.perf_counter() - t0, 3)
        print(f"{label}: {json.dumps(s)}")


if __name__ == "__main__":
    main()
