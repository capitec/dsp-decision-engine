# The pipeline of the spec's worked example, verbatim; the tree document is inline.
import json

import polars as pl

from decider import branch, dag, engine, flow, frame_step, missing_as, param, step
from decider.engine.ir.nodes import CallNode
from decider.steps.trees import TreeConfig

RISK_TREE = """
{"type": "tree", "name": "risk_tree", "tree": {
  "nodes": [
    {"id": "n0", "data": {"type": "unary", "condition":
      {"op": ">", "feature": "ratio", "threshold": {"param": "hi_thresh", "default": 0.7}}}},
    {"id": "n1", "data": {"type": "leaf", "result_idx": 0}}],
  "edges": [{"source": "n0", "target": "n1", "data": {"sourceIndex": 0}}],
  "output": {"data": [{"risk_band": 1}], "default": {"risk_band": 0}, "dtypes": [["risk_band", "Int64"]]}}}
"""


BUREAU = pl.DataFrame({"client_id": [1], "bureau_score": [700]})


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float, min_ratio: float = param(0.3, shared_key="min_ratio")) -> bool:
    return ratio >= min_ratio


affordability = dag(disposable_income, ratio, affordable, name="affordability")


def term_cap(requested_term: float, ceiling: float = param(60.0, ge=6, le=84)) -> float:
    return min(requested_term, ceiling)


@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float = missing_as(0.0),
                  cap: float = param(48.0, ge=6, le=60)) -> float:
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap


@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)


@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(outputs=("band", "band_score"))
def banding(ratio: float) -> tuple[int, float]:
    return (1, 10.0) if ratio > 2 else (0, 0.0)


term = flow(
    term_cap,
    cap_by_income,
    branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
    name="term",
)


@frame_step(reads=["client_id"], writes=["bureau_score"])
def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(BUREAU, on="client_id", how="left")


# Sources are import paths, so give the example's functions the module they'd live in.
for _fn in (disposable_income, ratio, affordable, term_cap, is_private, cap_by_income.fn, cap_private.fn,
            cap_public.fn, banding.fn, join_bureau.fn):
    _fn.__module__ = "app.pipeline"

risk_tree = TreeConfig.load(RISK_TREE)

pipeline = (join_bureau | affordability | banding | term | risk_tree).emit("term_cap@*")


WALK = """
join_bureau                 FrameStep
affordability               DagStep
affordability/disposable_income   FunctionStep
affordability/ratio         FunctionStep
affordability/affordable    FunctionStep
banding                     FunctionStep   outputs=band, band_score
term                        SequentialStep
term/term_cap               FunctionStep
term/cap_by_income          FunctionStep
term/by_sector              BranchStep
risk_tree                   TreeConfig
"""

IR = """
SequenceNode  <root>
  CallNode[frame]   join_bureau                   source=app.pipeline:join_bureau
  SequenceNode      affordability                 source=decider.steps:DagStep
    CallNode        affordability/disposable_income   source=app.pipeline:disposable_income
    CallNode        affordability/ratio           source=app.pipeline:ratio
    CallNode        affordability/affordable      source=app.pipeline:affordable
  CallNode          banding                       source=app.pipeline:banding
  SequenceNode      term                          source=decider.steps:SequentialStep
    CallNode        term/term_cap                 source=app.pipeline:term_cap
    CallNode        term/cap_by_income            source=app.pipeline:cap_by_income
    BranchNode      term/by_sector                source=decider.steps:BranchStep
      CallNode      term/by_sector/is_private
      CallNode      term/by_sector/cap_private
      CallNode      term/by_sector/cap_public
  CallNode[row]     risk_tree                     source=decider.steps.trees:TreeConfig  reference=yes
"""

DEFAULTS = """
{
  "shared": {"min_ratio": 0.3},
  "term": {
    "term_cap":      {"ceiling": 60.0},
    "cap_by_income": {"cap": 48.0},
    "by_sector": {"cap_private": {"cap": 54.0}, "cap_public": {"cap": 60.0}}
  },
  "risk_tree": {"hi_thresh": 0.7}
}
"""


def _listing(node, depth=0):
    label = type(node).__name__
    if isinstance(node, CallNode) and node.kind != "scalar":
        label += f"[{node.kind}]"
    yield depth, label, node.origin.path or "<root>", node
    for child in node.children():
        yield from _listing(child, depth + 1)


def test_walk_shows_the_authoring_tree():
    expected = [line.split(None, 2) for line in WALK.strip().splitlines()]
    actual = list(pipeline.walk())
    assert [(p, type(s).__name__) for p, s in actual] == [(e[0], e[1]) for e in expected]
    extras = {e[0]: e[2] for e in expected if len(e) == 3}
    assert extras == {"banding": "outputs=band, band_score"}
    assert dict(actual)["banding"].outputs == ("band", "band_score")


def test_to_ir_matches_the_listing_with_origins():
    lines = IR.strip("\n").splitlines()
    actual = list(_listing(engine.to_ir(pipeline)))
    assert len(actual) == len(lines)
    for line, (depth, label, path, node) in zip(lines, actual):
        tokens = line.split()
        assert (depth, label, path) == ((len(line) - len(line.lstrip())) // 2, tokens[0], tokens[1])
        extras = dict(t.split("=", 1) for t in tokens[2:])
        if "source" in extras:
            assert node.origin.source == extras["source"]
        if isinstance(node, CallNode):
            assert (node.reference is not None) == ("reference" in extras)
    root = actual[0][3]
    assert root.origin.source == "decider.steps:SequentialStep"
    assert root.emits == ("term_cap@*",)
    assert actual[-2][3].origin.source == "app.pipeline:cap_public"


def test_defaults_document_matches():
    assert pipeline.parameters().defaults() == json.loads(DEFAULTS)


def test_parameters_report_type_default_bounds_and_users():
    schema = pipeline.parameters()
    assert list(schema)[0] == "shared"
    assert schema["shared"] == {
        "min_ratio": {"type": "float", "default": 0.3, "used_by": ["affordability/affordable"]}
    }
    assert schema["term/cap_by_income"] == {"cap": {"type": "float", "default": 48.0, "ge": 6, "le": 60}}
    assert schema["risk_tree"] == {"hi_thresh": {"type": "float", "default": 0.7}}
    assert "affordability/affordable" not in schema


def test_json_schema_nests_params_by_path():
    schema = pipeline.parameters().json_schema()
    cap = schema["properties"]["term"]["properties"]["cap_by_income"]["properties"]["cap"]
    assert cap == {"default": 48.0, "maximum": 60, "minimum": 6, "type": "number"}
    assert schema["properties"]["shared"]["properties"]["min_ratio"]["default"] == 0.3
    assert schema["properties"]["risk_tree"]["properties"]["hi_thresh"] == {"type": "number", "default": 0.7}


def test_steps_are_still_plain_functions():
    assert cap_by_income(term_cap=60.0, min_net_salary=4000.0) == 48.0
    assert banding(ratio=3.0) == (1, 10.0)
