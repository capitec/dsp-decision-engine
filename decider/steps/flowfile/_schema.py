"""The flow-file shape: a root document and its step, flow, dag, branch, loop, each and optimise nodes."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExprCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expr: str
    feature_types: dict[str, str] | None = None


Condition = str | ExprCondition


class RootDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow: str
    components: str | None = None
    id: str | None = None
    suppress_checks: list[str] = []
    steps: list[dict[str, Any]] = Field(min_length=1)


class StepNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    as_: str | None = Field(None, alias="as")
    suppress_checks: list[str] = []


class FlowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow: str
    steps: list[dict[str, Any]] | None = None
    id: str | None = None
    suppress_checks: list[str] = []


class BranchNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: str
    when: Condition
    modifies: list[str]
    then: list[dict[str, Any]]
    else_: list[dict[str, Any]] | None = Field(None, alias="else")
    id: str | None = None
    suppress_checks: list[str] = []


class LoopNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loop: str
    while_: Condition = Field(alias="while")
    carries: list[str]
    max: int
    steps: list[dict[str, Any]]
    id: str | None = None
    suppress_checks: list[str] = []


class DagNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dag: str
    steps: list[dict[str, Any]]
    inputs: list[str] | None = None
    id: str | None = None
    suppress_checks: list[str] = []


class EachNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    each: str
    column: str
    steps: list[dict[str, Any]]
    output: str | None = None
    id: str | None = None
    suppress_checks: list[str] = []


class OptimiseNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimise: str
    count: list[dict[str, Any]]
    evaluate: list[dict[str, Any]]
    score: str = "score"
    disqualify: list[dict[str, Any]] | None = None
    max_candidates: int
    id: str | None = None
    suppress_checks: list[str] = []
