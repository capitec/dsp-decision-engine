# Tree formats
from .tree.v1.tree import Tree as V1Tree
from .tree.v2.tree import Tree as V2Tree
from .tree.v3.tree import Tree as V3Tree
from .tree.tree import Tree

# Flat rule module (the main execution unit)
from .flat_rules.module import (
    FlatRuleModule,
    PrioritizedFlatRuleModule,
    PrioritizationMode,
    RunPolarsExpression,
    OptimRunPolarsExpression,
)

# Rule node types
from .flat_rules.nodes import (
    LeafRule,
    UnaryRule,
    CompositeRule,
    CasesRanges,
    CasesStringMatch,
    CasesIsIn,
    CasesRule,
    RuleRoot,
    RuleMeta,
    FlatRuleTree,
    BuilderConfig,
    RuleType,
)

# Condition/operator primitives
from .common.nodes import (
    TUnaryOp,
    TCondition,
    CompositeCondition,
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
    CasesBranch,
    BaseUnaryNode,
    BaseCasesRanges,
    BaseCasesStringMatch,
    BaseCasesIsIn,
    BaseCompositeNode,
    UnaryLessThanEqual,
    UnaryLessThan,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryNotEqual,
    UnaryBetween,
    UnaryIsIn,
    UnaryStringMatch,
    UnaryIsNull,
    UnaryIsNotNull,
    UnaryIsTrue,
    UnaryIsFalse,
)

# Shared types
from .common.shared import InputRef, TreeOutput, WithTreeOutput
from .common.parameters import WithParameters, ParameterInfo
from .common.feature import Feature
from .common.nodetypes import (
    BaseRule,
    TLogicOp,
    TNodeType,
    TStringMatchType,
    TNullHandling,
    RangeEndLogic,
    NodeMeta,
    NodePosition,
)

# Serializable / schema types
# Discriminated union for all module types
from .modules import TModule, TTreeFormat, ModuleWrapper, module_discriminator


def register_rule_modules():
    """Register all rule module types into the global GraphModule union."""
    from decider.modules import register_graph_module

    for module_cls in [V1Tree, V2Tree, V3Tree, FlatRuleModule, PrioritizedFlatRuleModule]:
        register_graph_module(module_cls)


__all__ = [
    # Trees
    "V1Tree",
    "V2Tree",
    "V3Tree",
    "Tree",
    # Flat rules
    "FlatRuleModule",
    "PrioritizedFlatRuleModule",
    "PrioritizationMode",
    "RunPolarsExpression",
    "OptimRunPolarsExpression",
    # Rule nodes
    "LeafRule",
    "UnaryRule",
    "CompositeRule",
    "CasesRanges",
    "CasesStringMatch",
    "CasesIsIn",
    "CasesRule",
    "RuleRoot",
    "RuleMeta",
    "FlatRuleTree",
    "BuilderConfig",
    "RuleType",
    # Conditions & operators
    "TUnaryOp",
    "TCondition",
    "CompositeCondition",
    "RangeCondition",
    "StringMatchCondition",
    "IsInCondition",
    "CasesBranch",
    "BaseUnaryNode",
    "BaseCasesRanges",
    "BaseCasesStringMatch",
    "BaseCasesIsIn",
    "BaseCompositeNode",
    "UnaryLessThanEqual",
    "UnaryLessThan",
    "UnaryEqual",
    "UnaryGreaterThan",
    "UnaryGreaterThanEqual",
    "UnaryNotEqual",
    "UnaryBetween",
    "UnaryIsIn",
    "UnaryStringMatch",
    "UnaryIsNull",
    "UnaryIsNotNull",
    "UnaryIsTrue",
    "UnaryIsFalse",
    # Shared types
    "InputRef",
    "TreeOutput",
    "WithTreeOutput",
    "WithParameters",
    "ParameterInfo",
    "Feature",
    "BaseRule",
    "TLogicOp",
    "TNodeType",
    "TStringMatchType",
    "TNullHandling",
    "RangeEndLogic",
    "NodeMeta",
    "NodePosition",
    # Module union
    "TModule",
    "TTreeFormat",
    "ModuleWrapper",
    "module_discriminator",
    # Registration
    "register_rule_modules",
]
