from __future__ import annotations

from decikit._engine.ir.nodes import BranchNode, CallNode, IRNode, LoopNode, ScatterGatherNode


def interface(node: IRNode) -> tuple[set[str], set[str]]:
    """The names `node` reads from outside and the names it writes.

    A branch writes only its `modifies`, a loop only its `carries` and an
    `each` reads its list column and writes its output (its child's names are
    the element's fields, not the pipeline's); a frame step of unknown lineage reads and writes nothing known.

    Example::

        reads, writes = interface(to_ir(affordability))
        # ({"net_income", "expenses", "instalment"}, {"disposable_income", "ratio", "affordable"})
    """
    if isinstance(node, CallNode):
        return {i.name for i in node.inputs or ()}, {o.name for o in node.outputs or ()}
    if isinstance(node, BranchNode):
        reads, cond_writes = interface(node.condition)
        for arm in node.arms:
            reads |= interface(arm)[0] - cond_writes
        return reads, set(node.modifies)
    if isinstance(node, ScatterGatherNode):
        return {node.column}, {node.output}
    reads, writes = set(), set()
    for child in node.children():
        r, w = interface(child)
        reads |= r - writes
        writes |= w
    if isinstance(node, LoopNode):
        return reads | set(node.carries), set(node.carries)
    return reads, writes
