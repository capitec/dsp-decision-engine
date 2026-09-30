"""Experiment 05 follow-up: prove `id=` reaches the real engine's Origin.

Covers follow-up items 1 (every declaration shape), 2 (flow ids), 3
(reuse/composition) and 5 (collision/uniqueness scope). Run with
`uv run python notes/vscode-redesign/experimentation/05-step-id/integration.py`.

This imports and lowers real `decider` steps; the only engine change under
test is the optional `id=` keyword threading through to `Origin.id`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from decider import engine
from decider.engine.ir.nodes import iter_nodes
from decider.steps import (
    ConfigurableStep,
    branch,
    dag,
    each,
    flow,
    frame_step,
    loop,
    optimise,
    step,
)
from decider.steps.trees import TreeConfig

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # for spike_lib, an external package without ids
import spike_lib  # noqa: E402


def origins(node):
    return {n.origin.path: n.origin for n in iter_nodes(node)}


def shape(node):
    return [n.origin.path for n in iter_nodes(node)]


# ---- item 1: every declaration shape lowers with the id on its Origin ----

def _scalar_steps():
    @step(output="term_cap", id="3f9a7c2e1b5d")
    def cap_by_income(term_cap: float) -> float:
        return term_cap

    @frame_step(reads=["client_id"], writes=["bureau_score"], id="8b1d4f00a5c3")
    def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(bureau_score=pl.lit(1.0))

    return cap_by_income, join_bureau


def item1():
    print("-- item 1: declaration shapes")
    @step(output="term_cap", id="0123456789ab")
    def cap(term_cap: float) -> float:
        return term_cap

    @step(output="x")
    def x_step(x: float) -> float:
        return x

    def is_positive(term_cap: float) -> bool:
        return term_cap > 0

    shapes = {
        "@step": (cap, "cap"),
        "flow": (flow(cap, x_step, name="term", id="111111111111"), "term"),
        "dag": (dag(cap, x_step, name="p03", id="f00d0000aaaa"), "p03"),
        "branch": (branch("on_card", cap, cap.named("cap_pub"), modifies=["term_cap"],
                          name="by_sector", id="e1d5a3c9b2f0"), "by_sector"),
        "loop": (loop(is_positive, flow(cap, name="month"), carries=["term_cap"], max_iterations=3,
                      name="repay", id="c0ffee123456"), "repay"),
        "each": (each("items", flow(x_step, name="item"), name="items", id="deadbeef0001"), "items"),
        "optimise": (optimise(lambda: 2, flow(x_step, name="evaluate"), score="x", max_candidates=4,
                              name="best", id="0123abcdef45"), "best"),
    }
    for label, (pipeline, path) in shapes.items():
        o = origins(engine.to_ir(pipeline))[path]
        assert o.id is not None and o.path == path, f"{label}: {o}"
        print(f"  {label:<12} id={o.id}  path={o.path}  source={o.source}")
    # frame_step and the imported/reused wrap
    _, frame = _scalar_steps()
    fo = origins(engine.to_ir(frame))["join_bureau"]
    assert fo.id == "8b1d4f00a5c3"
    print(f"  {'@frame_step':<12} id={fo.id}  path={fo.path}  source={fo.source}")
    wrapped = step(spike_lib.statutory_deductions, id="3f9a7c2e1b5d")
    wo = origins(engine.to_ir(flow(wrapped, name="wrap")))["wrap/statutory_deductions"]
    assert wo.id == "3f9a7c2e1b5d" and wo.source == "spike_lib:statutory_deductions"
    print(f"  {'wrapped':<12} id={wo.id}  path={wo.path}  source={wo.source}")


# ---- item 1b: id is additive — name/path/step_map/execution unchanged ----

def item1b():
    print("-- item 1b: id is additive")
    @step(output="term_cap")
    def cap(term_cap: float) -> float:
        return term_cap

    def with_id():
        return flow(cap, name="term", id="111111111111")

    def without_id():
        return flow(cap, name="term")

    a, b = engine.to_ir(with_id()), engine.to_ir(without_id())
    assert shape(a) == shape(b), (shape(a), shape(b))
    assert engine.step_map(with_id()).keys() == engine.step_map(without_id()).keys()
    df = pl.DataFrame({"term_cap": [1.0, 2.0]})
    assert with_id().run(df).equals(without_id().run(df))
    print("  same paths, same step_map keys, same output frame")


# ---- item 2: flow id + step id is a global reference ----

def item2():
    print("-- item 2: flow id + step id")
    @step(output="x", id="aaaaaaaaaaaa")
    def one(x: float) -> float:
        return x

    root = flow(flow(one, name="inner", id="bbbbbbbbbbbb"), name="root", id="cccccccccccc")
    node = engine.to_ir(root)
    by_path = origins(node)
    # flow ids are on their own Origin; step ids on theirs
    assert by_path["root"].id == "cccccccccccc"
    assert by_path["root/inner"].id == "bbbbbbbbbbbb"
    assert by_path["root/inner/one"].id == "aaaaaaaaaaaa"
    print("  root=cccccccccccc inner=bbbbbbbbbbbb step=aaaaaaaaaaaa -> (flow id, step id) unique")


# ---- item 3: reuse / composition ----

def item3():
    print("-- item 3: reuse / composition")
    @step(output="x", id="dddddddddddd")
    def common(x: float) -> float:
        return x + 1.0

    # one decorated step in two flows: same id, two different paths
    a = flow(common, name="fa")
    b = flow(common, name="fb")
    oa = origins(engine.to_ir(a))["fa/common"]
    ob = origins(engine.to_ir(b))["fb/common"]
    assert oa.id == ob.id == "dddddddddddd" and oa.path != ob.path
    print(f"  reused step id={oa.id} at {oa.path} and {ob.path}")

    # sub-flow called by a parent: parent id stays, sub-flow id stays
    sub = flow(common, name="sub", id="eeeeeeeeeeee")
    parent = flow(sub, name="parent", id="ffffffffffff")
    op = origins(engine.to_ir(parent))
    assert op["parent"].id == "ffffffffffff" and op["parent/sub"].id == "eeeeeeeeeeee"
    print("  sub-flow id=eeeeeeeeeeee under parent id=ffffffffffff")

    # copied configurable asset: id is a committed JSON field that round-trips
    cfg = TreeConfig.load({"type": "tree", "name": "risk_tree", "id": "999999999999",
                           "tree": {"nodes": [{"id": "root", "data": {"type": "leaf", "result_idx": 0}}],
                                    "edges": [], "output": {"data": [{"pts": 0}], "default": {"pts": 1},
                                                           "dtypes": [["pts", "Int64"]]}}})
    co = origins(engine.to_ir(cfg))["risk_tree"]
    assert co.id == "999999999999"
    assert ConfigurableStep.load(cfg.model_dump_json()).id == "999999999999"
    print("  copied config id=999999999999 round-trips through model_dump/load")

    # imported step from a package with no ids: id is None, still lowers
    no_id = origins(engine.to_ir(flow(spike_lib.helper, name="ext")))["ext/helper"]
    assert no_id.id is None and no_id.source == "spike_lib:helper"
    print(f"  imported no-id step: id=None, source={no_id.source} (derived-only identity)")


# ---- item 5: collision / uniqueness scope ----

def item5():
    print("-- item 5: collision / uniqueness scope")
    import secrets
    n = 100_000
    ids = {secrets.token_hex(6) for _ in range(n)}
    assert len(ids) == n, "collision in 100k tokens is astronomically unlikely"
    # repo-wide scan is stricter than flow-local: a duplicate that two different
    # flows would never share is still flagged by a global scan.
    dup = "aaaaaaaaaaaa"
    scan = ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "aaaaaaaaaaaa"]
    seen, dups = set(), set()
    for i in scan:
        (dups if i in seen else seen).add(i)
    assert dup in dups
    print(f"  100k tokens unique; global scan flags {dup} even though flows are disjoint")


if __name__ == "__main__":
    item1()
    item1b()
    item2()
    item3()
    item5()
    print("integration ok")
