from __future__ import annotations

import inspect
from typing import Any

from decider.contract.refs import StepRef
from decider.engine.ir.nodes import (BranchNode, CallNode, FiniteStateMachineNode, IRNode, LoopNode,
                                     ScatterGatherNode, SequenceNode, iter_nodes)

_NODE_TYPE = {
    CallNode: "call",
    SequenceNode: "sequence",
    BranchNode: "branch",
    LoopNode: "loop",
    ScatterGatherNode: "scatter_gather",
    FiniteStateMachineNode: "state_machine",
}


def function_source(fn: Any) -> tuple[str, int] | None:
    try:
        return inspect.getsource(fn), inspect.getsourcelines(fn)[1]
    except (OSError, TypeError):
        return None


def step_ref(node: IRNode, flow_id: str | None) -> StepRef:
    origin = node.origin
    return StepRef(
        flow_id=flow_id,
        step_id=origin.id,
        path=origin.path,
        name=origin.path.rpartition("/")[2],
        source=origin.source,
        locator=origin.locator,
        node_type=_NODE_TYPE.get(type(node)),
    )


def call_nodes(root: IRNode) -> list[CallNode]:
    return [n for n in iter_nodes(root) if isinstance(n, CallNode)]


def helper_calls(root: IRNode) -> set[int]:
    # The condition/transition/accumulate call of a composite is part of its
    # parent's identity, so it is never independently id'd.
    helpers: set[int] = set()
    for node in iter_nodes(root):
        if isinstance(node, BranchNode):
            helpers.add(id(node.condition))
        elif isinstance(node, LoopNode):
            helpers.add(id(node.condition))
        elif isinstance(node, ScatterGatherNode) and node.accumulate is not None:
            helpers.add(id(node.accumulate))
        elif isinstance(node, FiniteStateMachineNode):
            helpers.add(id(node.transition_function))
    return helpers
