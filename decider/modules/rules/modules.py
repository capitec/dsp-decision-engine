"""Unified module system for DSP Decider.

Combines tree-based and flat rule-based execution modules.
"""

import typing as t
from pydantic import RootModel, Field, Discriminator, Tag

# Tree modules (v1, v2, v3)
from .tree.v1.tree import Tree as V1Tree
from .tree.v2.tree import Tree as V2Tree
from .tree.v3.tree import Tree as V3Tree

# Flat rule module
from .flat_rules.module import PrioritizedFlatRuleModule


# =============================================================================
# Discriminated Union Types
# =============================================================================


# Union of all tree formats (v1, v2, v3)
TTreeFormat = t.Union[V1Tree, V2Tree, V3Tree]

# Union of all executable module types (tree formats + flat rules) — discriminated by type literal
TModule = t.Annotated[
    t.Union[V1Tree, V2Tree, V3Tree, PrioritizedFlatRuleModule],
    Field(discriminator="type"),
]


def module_discriminator(value: t.Any) -> str:
    """Return the type discriminator string for a module value."""
    if isinstance(value, dict):
        return value.get("type", "v3-tree")
    return getattr(value, "type", "v3-tree")


class ModuleWrapper(RootModel):
    """Wrapper for discriminated module union (tree or flat rule)."""

    root: TModule


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Tree versions
    "V1Tree",
    "V2Tree",
    "V3Tree",
    # Flat rules
    "PrioritizedFlatRuleModule",
    # Type unions
    "TTreeFormat",
    "TModule",
    "ModuleWrapper",
]
