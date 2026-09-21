"""Tree component - supports v1, v2, and v3 tree formats.

For module types that combine trees and flat rules, see dspd.components.modules
"""

import typing as t
from pydantic import Field

from .tree import Tree, _Tree
from .v1.tree import Tree as V1Tree
from .v2.tree import Tree as V2Tree
from .v3.tree import Tree as V3Tree

# Import flat rule module for TTree union
from ..flat_rules.module import PrioritizedFlatRuleModule

# Legacy alias
UiTree = V2Tree

# TTree union for API/DB usage - tree formats + flat rules
# Note: Uses _Tree (the internal union) + PrioritizedFlatRuleModule
TTree = t.Union[_Tree, PrioritizedFlatRuleModule]

__all__ = [
    "Tree",
    "V1Tree",
    "V2Tree",
    "V3Tree",
    "UiTree",
    "TTree",
]
