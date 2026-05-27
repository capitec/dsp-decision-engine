import typing as t

from decider.modules.expression import ExpressionModule, Node


class UnionExpressionModule(ExpressionModule):
    """Merges multiple ExpressionModules into a single compilation pass.

    All children's nodes are flattened into one CompiledExpressions artifact,
    so the combined module applies in a single frame pass.

    Created via the & operator:  mod_a & mod_b
    """

    type: t.Literal["union"]
    modules: t.List[ExpressionModule]

    def expand_nodes(self) -> t.Dict[str, Node]:
        merged: t.Dict[str, Node] = {}
        for mod in self.modules:
            merged.update(mod.expand_nodes())
        return merged
