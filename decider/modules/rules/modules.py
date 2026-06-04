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


def module_discriminator(value: t.Any) -> str:
    """Custom discriminator for nested tree/flat-rule union.

    Returns:
        - "v1_tree" for format_version=1
        - "v2_tree" for format_version=2
        - "v3_tree" for format_version=3
        - "flat_rule" for type="prioritized_flat_rule"
    """
    if isinstance(value, dict):
        # Check if it's a flat rule module
        if value.get("type") == "prioritized_flat_rule":
            return "flat_rule"

        # Otherwise, discriminate by format_version
        format_version = value.get("formatVersion") or value.get("format_version")
        if format_version == 1:
            return "v1_tree"
        elif format_version == 2:
            return "v2_tree"
        elif format_version == 3:
            return "v3_tree"

    # Fallback for objects (already instantiated models)
    if hasattr(value, "type") and value.type == "prioritized_flat_rule":
        return "flat_rule"

    if hasattr(value, "format_version"):
        if value.format_version == 1:
            return "v1_tree"
        elif value.format_version == 2:
            return "v2_tree"
        elif value.format_version == 3:
            return "v3_tree"

    # Default fallback (shouldn't reach here)
    return "v2_tree"


# Union of all tree formats (v1, v2, v3) - kept for backward compatibility
TTreeFormat = t.Annotated[
    t.Union[V1Tree, V2Tree, V3Tree], Field(discriminator="format_version")
]

# Union of all executable module types (tree formats + flat rules)
TModule = t.Annotated[
    t.Union[
        t.Annotated[V1Tree, Tag("v1_tree")],
        t.Annotated[V2Tree, Tag("v2_tree")],
        t.Annotated[V3Tree, Tag("v3_tree")],
        t.Annotated[PrioritizedFlatRuleModule, Tag("flat_rule")],
    ],
    Discriminator(module_discriminator),
]


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
