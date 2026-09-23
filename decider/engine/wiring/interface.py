from __future__ import annotations

from decider.engine.ir.nodes import BranchNode, CallNode, IRNode, LoopNode


def interface(node: IRNode) -> tuple[set[str], set[str]]:
    """The names `node` reads from outside and the names it writes.

    A branch writes only its `modifies` and a loop only its `carries`; a
    frame step of unknown lineage reads and writes nothing known.

    Example::

        reads, writes = interface(engine.to_ir(affordability))
        # ({"net_income", "expenses", "instalment"}, {"disposable_income", "ratio", "affordable"})
    """
    if isinstance(node, CallNode):
        return {i.name for i in node.inputs or ()}, {o.name for o in node.outputs or ()}
    if isinstance(node, BranchNode):
        reads, cond_writes = interface(node.condition)
        for arm in node.arms:
            reads |= interface(arm)[0] - cond_writes
        return reads, set(node.modifies)
    reads, writes = set(), set()
    for child in node.children():
        r, w = interface(child)
        reads |= r - writes
        writes |= w
    if isinstance(node, LoopNode):
        return reads | set(node.carries), set(node.carries)
    return reads, writes
