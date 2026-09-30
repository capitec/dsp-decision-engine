from __future__ import annotations

import importlib
import importlib.util
import inspect
import warnings
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from decider.contract.refs import CONTRACT_VERSION, EdgeRef, FlowDescription, FlowRef, StepRef, ValueSlotRef
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import (BranchNode, CallNode, FiniteStateMachineNode, IRNode, LoopNode, ScatterGatherNode,
                                     SequenceNode, iter_nodes)
from decider.engine.run.engine import RUNNERS
from decider.engine.wiring import Plan, resolve

_NODE_TYPE = {
    CallNode: "call",
    SequenceNode: "sequence",
    BranchNode: "branch",
    LoopNode: "loop",
    ScatterGatherNode: "scatter_gather",
    FiniteStateMachineNode: "state_machine",
}

# Optional extra (pyproject) -> the module that proves it is installed.
_OPTIONAL_EXTRAS = {
    "visualise": "graphviz",
    "serve-starlette": "uvicorn",
    "serve-sanic": "sanic",
    "notebook": "IPython",
    "eval": "simpleeval",
}


class Capabilities(BaseModel):
    """What this decider build can do, for callers that must degrade gracefully.

    `modes` lists the execution modes `Engine.bind` accepts; a mode outside it
    raises `EngineError`. `optional_dependencies` maps each optional extra to
    whether it is installed (unavailable ones raise
    `DeciderMissingDependencyError` on use). `source_mapping` says whether an
    import-path source can be resolved to a file and line.

    Example::

        capabilities().modes  # ("fused", "interpreted", "stepped")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: str = CONTRACT_VERSION
    modes: tuple[str, ...]
    optional_dependencies: dict[str, bool]
    source_mapping: bool = True


def capabilities() -> Capabilities:
    """The capabilities of the installed decider."""
    return Capabilities(
        modes=tuple(sorted(RUNNERS)),
        optional_dependencies={extra: importlib.util.find_spec(module) is not None
                               for extra, module in _OPTIONAL_EXTRAS.items()},
    )


def describe(step: Any) -> FlowDescription:
    """The static structure of `step` as a `FlowDescription`.

    Layers over the IR and wiring: nodes and control edges come from the IR
    tree, data edges and value slots from `resolve` (which knows which node
    wrote each value). A wiring problem raises the engine's own `WiringError`.

    Example::

        desc = describe(pipeline)
        [n.path for n in desc.nodes]
        [e.values for e in desc.edges if e.kind == "data"]
    """
    ir = to_ir(step)
    plan = resolve(ir)
    flow = FlowRef(name=ir.origin.path, source=ir.origin.source)
    return FlowDescription(
        flow=flow,
        nodes=tuple(_node_ref(n) for n in iter_nodes(ir)),
        edges=tuple(_control_edges(ir)) + tuple(_data_edges(plan)),
        value_slots=tuple(ValueSlotRef(name=v.name, path=v.producer) for v in plan.versions),
    )


def resolve_step(description: FlowDescription, ref: StepRef) -> StepRef:
    """The node of `description` that `ref` names.

    A durable reference matches by `(flow_id, step_id)`; when neither is set
    (or none matches) it falls back to the derived `path`. A durable reference
    that matches nothing warns and returns `ref` unchanged, so its capture-time
    name, source and path survive for repair instead of vanishing.

    Example::

        node = resolve_step(desc, StepRef(step_id="cap", path="term/cap_by_income", source="credit.rules:cap_by_income"))
    """
    if ref.step_id is not None:
        for node in description.nodes:
            if node.flow_id == ref.flow_id and node.step_id == ref.step_id:
                return node
        warnings.warn(f"durable reference {ref.flow_id}/{ref.step_id} resolves nowhere; "
                      f"kept {ref.path!r} ({ref.source})", stacklevel=2)
        return ref
    for node in description.nodes:
        if node.path == ref.path:
            return node
    return ref


def source_location(source: str) -> tuple[str, int] | None:
    """The `(file, line)` an import-path `source` points at, or `None` when it can't be resolved.

    A step whose module can't be imported, or that is built in, has no source
    location; callers treat that as unavailable rather than an error.

    Example::

        source_location("credit.rules:cap_by_income")  # ("credit/rules.py", 41)
    """
    module_name, _, qualname = source.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return (inspect.getsourcefile(obj) or "", inspect.getsourcelines(obj)[1])
    except (ImportError, AttributeError, TypeError, OSError):
        return None


def _node_ref(node: IRNode) -> StepRef:
    origin = node.origin
    return StepRef(
        path=origin.path,
        name=origin.path.rpartition("/")[2],
        source=origin.source,
        locator=origin.locator,
        node_type=_NODE_TYPE[type(node)],
        kind=node.kind if isinstance(node, CallNode) else None,
        inputs=tuple(i.name for i in node.inputs or ()) if isinstance(node, CallNode) else (),
        outputs=tuple(o.name for o in node.outputs or ()) if isinstance(node, CallNode) else (),
    )


def _control_edges(root: IRNode) -> list[EdgeRef]:
    return [EdgeRef(from_path=node.origin.path, to_path=child.origin.path, kind="control")
            for node in iter_nodes(root) for child in node.children()]


def _data_edges(plan: Plan) -> list[EdgeRef]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for call in plan.calls:
        for read in call.reads or ():
            if read.producer is None or read.producer == call.node.origin.path:
                continue
            grouped.setdefault((read.producer, call.node.origin.path), set()).add(read.name)
    return [EdgeRef(from_path=src, to_path=dst, kind="data", values=tuple(sorted(names)))
            for (src, dst), names in sorted(grouped.items())]
