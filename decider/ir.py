"""The intermediate representation of a flow: declarations, nodes and origins.

Example::

    from decider.ir import iter_nodes, to_ir
    paths = [node.path for node in iter_nodes(to_ir(pipeline))]
"""
from decider._engine.ir import (BranchNode, CallNode, Input, IRContext, IRNode, LoopNode, NullPolicy, Origin, Output,
                                ParamDecl, SequenceNode)
from decider._engine.ir.context import step_map, to_ir
from decider._engine.ir.nodes import iter_nodes

__all__ = [
    "BranchNode", "CallNode", "IRContext", "IRNode", "Input", "LoopNode", "NullPolicy", "Origin", "Output",
    "ParamDecl", "SequenceNode", "iter_nodes", "step_map", "to_ir",
]
