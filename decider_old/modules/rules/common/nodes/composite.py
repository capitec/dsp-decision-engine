"""Base composite node — AND/OR/NOT logic + required feature/param traversal.

Both flat_rules.CompositeRule and nodes_ui.CompositeNode extend this.
"""

import typing as t
from pydantic import Field, model_validator
import typing_extensions as t_ext

from ..nodetypes import BaseRule, TLogicOp
from .conditions import TCondition


class BaseCompositeNode(BaseRule):
    """Composite AND/OR/NOT node — base shared between systems."""

    type: t.Literal["composite"] = "composite"
    id: t.Optional[str] = None
    op: TLogicOp
    conditions: t.List[TCondition] = Field(
        description="Conditions to combine with AND/OR/NOT"
    )

    @model_validator(mode="after")
    def validate_conditions(self) -> t_ext.Self:
        if self.op == TLogicOp.NOT and len(self.conditions) != 1:
            raise ValueError("NOT operator must have exactly 1 condition")
        return self

    def get_required_features(self) -> t.Set[str]:
        features = set()
        for cond in self.conditions:
            features.update(cond.get_required_features())
        return features

    def get_required_parameters(self) -> t.Set[str]:
        params = set()
        for cond in self.conditions:
            params.update(cond.get_required_parameters())
        return params
