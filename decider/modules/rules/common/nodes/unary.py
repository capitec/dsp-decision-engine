"""Base unary node — condition + required feature/param traversal.

Both flat_rules.UnaryRule and nodes_ui.UnaryNode extend this.
They add their own child-resolution mechanism (embedded vs edge-based).
"""

import typing as t
from pydantic import Field

from dspd.components.common.nodetypes import BaseRule
from dspd.components.common.nodes.operators import TUnaryOp


class BaseUnaryNode(BaseRule):
    """Unary condition node — holds the condition, delegates child logic to subclasses."""

    type: t.Literal["unary"] = "unary"
    id: t.Optional[str] = Field(default=None)
    condition: TUnaryOp = Field(description="The condition to evaluate")

    def get_required_features(self) -> t.Set[str]:
        return self.condition.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.condition.get_required_parameters()
