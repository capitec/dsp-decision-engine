"""The intermediate representation of a flow: declarations, nodes and origins.

Example::

    from decikit.ir import iter_nodes, to_ir
    paths = [node.path for node in iter_nodes(to_ir(pipeline))]
"""
from decikit._engine.ir import (BranchNode, CallNode, Input, IRContext, IRNode, LoopNode, NullPolicy, Origin, Output,
                                ParamDecl, SequenceNode)
from decikit._engine.ir.context import step_map, to_ir
from decikit._engine.ir.absent import has_optional_key, without_absent
from decikit._engine.ir.decls import base_annotation
from decikit._engine.ir.nodes import FiniteStateMachineNode, ScatterGatherNode, iter_nodes
from decikit._engine.ir.origin import check_id, check_name
from decikit._engine.types import decimal_info, pydantic_ready

__all__ = [
    "BranchNode", "CallNode", "FiniteStateMachineNode", "IRContext", "IRNode", "Input", "LoopNode", "NullPolicy",
    "Origin", "Output", "ParamDecl", "ScatterGatherNode", "SequenceNode", "base_annotation", "check_id", "check_name",
    "decimal_info", "has_optional_key", "iter_nodes", "pydantic_ready", "step_map", "to_ir", "without_absent",
]
