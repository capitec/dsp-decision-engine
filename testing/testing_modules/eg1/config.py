import typing as t
from ped.modules.core import BaseModule, PEDNode
from .impl import (
    avg_3wk_spend,
    acquisition_cost,
    spend_mean,
    spend_zero_mean,
    spend_std_dev,
    spend_zero_mean_unit_variance,
)


class EG1Module(BaseModule):
    """Example module 1 - spend analysis functions."""
    type: t.Literal["eg1"]
    
    def expand_nodes(self) -> t.List[PEDNode]:
        return [
            PEDNode.from_callable(avg_3wk_spend),
            PEDNode.from_callable(acquisition_cost),
            PEDNode.from_callable(spend_mean),
            PEDNode.from_callable(spend_zero_mean),
            PEDNode.from_callable(spend_std_dev),
            PEDNode.from_callable(spend_zero_mean_unit_variance),
        ]