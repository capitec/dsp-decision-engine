import typing as t
from hamilton import node
from decider.dag.core import DeciderExpandableModule
from . import funcs

class EG1Module(DeciderExpandableModule):
    def expand_nodes(self) -> t.Dict[str, node.Node]:
        return {
            "avg_3wk_spend": node.Node.from_fn(funcs.avg_3wk_spend),
            "acquisition_cost": node.Node.from_fn(funcs.acquisition_cost),
            "spend_mean": node.Node.from_fn(funcs.spend_mean),
            "spend_zero_mean": node.Node.from_fn(funcs.spend_zero_mean),
            "spend_std_dev": node.Node.from_fn(funcs.spend_std_dev),
            "spend_zero_mean_unit_variance": node.Node.from_fn(funcs.spend_zero_mean_unit_variance),
        }
    def compile(self):
        from decider.dag.compile import CompiledModulePlaceholder
        # TODO do this from the nodes and maybe a wrapper or something?
        return CompiledModulePlaceholder(funcs)
