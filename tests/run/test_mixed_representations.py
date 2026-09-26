"""Steps that read one column different ways: which combinations compose, and which are refused."""
from __future__ import annotations

import warnings
from typing import TypedDict

import polars as pl
import pytest

from decider import Engine, Raw, Rows, Struct, dag, flow, raw_str, step
from decider.exceptions import FallbackWarning, WiringError
from decider.steps.trees import TreeConfig
from decider.testing import assert_equivalent

PRIVATE = raw_str("private")
FRAME = pl.DataFrame(
    {"sector": ["private", "public"],
     "accounts": [[{"balance": 1.0}], [{"balance": 2.0}]],
     "applicant": [{"income": 5.0}, {"income": 6.0}]},
    schema_overrides={"applicant": pl.Struct({"income": pl.Float64})},
)


class Account(TypedDict):
    balance: float


class Applicant(TypedDict):
    income: float


def semantic(sector: str) -> float:
    return 1.0 if sector == "private" else 0.0


def code(sector: Raw[str]) -> float:
    return 10.0 if sector == PRIVATE else 0.0


def span(sector: Raw[bytes]) -> float:
    return 100.0 if sector == "private" else 0.0


def semantic_bytes(sector: bytes) -> float:
    return 1000.0 if sector == b"private" else 0.0


def listed(accounts: list[dict]) -> float:
    return sum(a["balance"] for a in accounts)


def rowed(accounts: Rows[Account]) -> float:
    total = 0.0
    for j in range(len(accounts.balance)):
        total += accounts.balance[j]
    return total


def dicted(applicant: dict) -> float:
    return applicant["income"]


def structed(applicant: Struct[Applicant]) -> float:
    return applicant["income"] * 2


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FallbackWarning)
        yield


# --- the same type, two representations: these compose ------------------------------------------

def test_a_semantic_str_and_a_raw_str_read_the_same_column_in_one_flow():
    # One runs in Python and one in the kernel, over the same version of the column.
    out = assert_equivalent(flow(semantic, code, name="p"), FRAME)
    assert (out["semantic"].to_list(), out["code"].to_list()) == ([1.0, 0.0], [10.0, 0.0])


def test_a_semantic_str_and_a_raw_str_read_the_same_column_in_a_dag():
    out = assert_equivalent(dag(step(semantic), step(code), name="p"), FRAME)
    assert (out["semantic"].to_list(), out["code"].to_list()) == ([1.0, 0.0], [10.0, 0.0])


def test_a_semantic_bytes_and_a_raw_bytes_read_the_same_column():
    out = assert_equivalent(flow(semantic_bytes, span, name="p"), FRAME)
    assert (out["semantic_bytes"].to_list(), out["span"].to_list()) == ([1000.0, 0.0], [100.0, 0.0])


def test_each_representation_of_one_column_lands_where_it_belongs():
    exe = Engine().bind(flow(semantic, code, name="p"), mode="fused")
    exe.run(FRAME)
    # The semantic reader runs in Python; the code reader stays in the kernel around it.
    assert set(exe.fallbacks()) == {"p/semantic"}


def test_a_raw_str_step_reads_a_semantic_str_step_s_output():
    def label(sector: str) -> str:
        return sector

    def measure(label: Raw[str]) -> float:
        return 10.0 if label == PRIVATE else 0.0

    out = assert_equivalent(flow(label, measure, name="p"), FRAME)
    assert out["measure"].to_list() == [10.0, 0.0]


# --- different types on one column: refused, naming both readers --------------------------------

REFUSED = [
    (semantic, span, "as str", "as bytes"),
    (code, span, "as str", "as bytes"),
    (listed, rowed, "as list", "as Rows"),
    (dicted, structed, "as dict", "as Struct"),
]


@pytest.mark.parametrize("first, second, one, other", REFUSED,
                         ids=[f"{a.__name__}+{b.__name__}" for a, b, *_ in REFUSED])
def test_two_readers_of_one_column_must_want_the_same_type(first, second, one, other):
    with pytest.raises(WiringError, match=f"{one}.*{other}"):
        Engine().bind(flow(first, second, name="p"), mode="fused")


@pytest.mark.parametrize("first, second, one, other", REFUSED,
                         ids=[f"{a.__name__}+{b.__name__}" for a, b, *_ in REFUSED])
def test_the_same_clash_is_refused_in_a_dag_and_in_interpreted_mode(first, second, one, other):
    for maker in (lambda: flow(first, second, name="p"), lambda: dag(step(first), step(second), name="p")):
        with pytest.raises(WiringError, match=f"{one}.*{other}"):
            Engine().bind(maker(), mode="interpreted")


# --- a decision tree reads its string features as bytes -----------------------------------------

HIT = {"data": [{"hit": 0}, {"hit": 1}], "default": {"hit": -1}, "dtypes": [["hit", "Int64"]]}
GATE = TreeConfig(name="gate", tree={
    "nodes": [{"id": "root", "data": {"type": "unary", "condition": {
                  "op": "string_match", "feature": "sector", "patterns": ["private"], "match_type": "exact"}}},
              {"id": "yes", "data": {"type": "leaf", "result_idx": 1}},
              {"id": "no", "data": {"type": "leaf", "result_idx": 0}}],
    "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}},
              {"source": "root", "target": "no", "data": {"sourceIndex": 1}}],
    "output": HIT})


def test_a_string_gated_tree_runs_on_its_own():
    assert assert_equivalent(flow(GATE, name="p"), FRAME)["hit"].to_list() == [1, 0]


def test_a_tree_and_a_raw_bytes_step_share_the_string_column():
    out = assert_equivalent(flow(GATE, span, name="p"), FRAME)
    assert (out["hit"].to_list(), out["span"].to_list()) == ([1, 0], [100.0, 0.0])


def test_a_tree_and_a_semantic_str_step_on_one_column_are_refused():
    # A tree matches bytes; a `str` step wants the Python string. The column has one type.
    with pytest.raises(WiringError, match="as bytes.*as str|as str.*as bytes"):
        Engine().bind(flow(GATE, semantic, name="p"), mode="fused")
