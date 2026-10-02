"""The flow-file shape: a root document and its step/flow/branch/loop nodes."""
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
    steps: list[dict[str, Any]] = Field(min_length=1)


class StepNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    as_: str | None = Field(None, alias="as")


class FlowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow: str
    steps: list[dict[str, Any]] | None = None
    id: str | None = None


class BranchNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: str
    when: Condition
    modifies: list[str]
    then: list[dict[str, Any]]
    else_: list[dict[str, Any]] | None = Field(None, alias="else")
    id: str | None = None


class LoopNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loop: str
    while_: Condition = Field(alias="while")
    carries: list[str]
    max: int
    steps: list[dict[str, Any]]
    id: str | None = None


NODE_MODELS: dict[str, type[BaseModel]] = {"step": StepNode, "flow": FlowNode, "branch": BranchNode, "loop": LoopNode}
