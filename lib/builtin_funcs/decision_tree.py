import typing
import pandas as pd
from lib import helpers
from itertools import chain
from dataclasses import dataclass
from uuid import uuid4

# TODO maybe make an actual data structure
_WhenTree = typing.List[
    typing.Union[
        typing.Tuple[str, typing.Union[str, "_WhenTree"]],
        str
    ]
]

@dataclass
class ConditionedNode:
    child: typing.Union[str, "WhenTree"]
    condition_key: typing.Optional[str] = None


@dataclass
class ExprCondition:
    expr: str

@dataclass
class ValueCondition:
    value: str

@dataclass
class WhenTree:
    nodes: typing.List[ConditionedNode]


    @classmethod
    def from_config(cls, config: dict) -> typing.Tuple["WhenTree", typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]:
        """
        Create a : class : ` WhenTree ` from a configuration. This is a convenience method for creating a when tree from the following configuration:
        when {
            expr = "x>3"
            value = 10
        }
        when {
            param = boolean_input
            when {
                value = 4
            }
            otherwise {
                value = 5
            }
        }
        otherwise {
            value = 10
        }
        Args:
            cls: The class to create from a configuration
            config: The configuration to use for creating the tree
        
        Returns: 
            A tuple containing the tree a dictionary of expressions and
        """

        # Get the list of when and otherwise statements as node_configs
        node_configs = config.get("when", [])
        if isinstance(node_configs, dict): node_configs = [node_configs]
        node_configs = list(node_configs) # Make a copy

        default = config.get("otherwise")
        if default is not None:
            assert isinstance(default, dict), "There can only be one default statement in a when tree"
            assert "expr" not in default, "Cannot have any expressions in default"
            assert "param" not in default, "Cannot have any parameters in default"
            node_configs.append(default) # Default value always at the end

        value_map = {}
        condition_map = {}
        nodes = []
        for node_conf in node_configs:
            # Extract the expression or parameter used for conditions
            expr = node_conf.get("expr")
            param = node_conf.get("param")
            condition_key = None
            if expr is not None and param is not None:
                raise ValueError("A when statement can only have a parameter or expression")
            if expr is not None:
                condition_key = str(uuid4())
                condition_map[condition_key] = ExprCondition(expr)
            if param is not None:
                condition_key = str(uuid4())
                condition_map[condition_key] = ValueCondition(param)

            # determine if it is a node type (direct value)
            value = helpers.infer_inject_parameter(node_conf, raise_on_fail=False)

            if value is not None:
                assert "when" not in node_conf and "otherwise" not in node_conf, "A when statement can either contain nested statements or a value but not both"
                node = ConditionedNode(child=str(uuid4()), condition_key=condition_key)
                value_map[node.child] = value
                nodes.append(node)
            else:
                child_when_tree, child_value_map, child_cond_map = cls.from_config(node_conf)
                assert len(child_when_tree.nodes) > 0, f"Must have at least one where or otherwise or value statement.\n{node_conf}"
                value_map.update(child_value_map)
                condition_map.update(child_cond_map)
                nodes.append(ConditionedNode(
                    child=child_when_tree,
                    condition_key=condition_key
                ))
        return cls(nodes=nodes), value_map, condition_map


def _vec_when_tree_rec(
        tree: WhenTree,
        value: typing.Union[pd.Series, pd.DataFrame], 
        variables: typing.Dict[str,typing.Union[pd.Series, pd.DataFrame]], 
        conditions: typing.Dict[str,pd.Series], 
        selected_subset: typing.Optional[pd.Series] = None
):
    # Done in reverse as earliest nodes take presence
    if selected_subset is None:
        selected_subset = pd.Series([True]*len(value))
    for node in reversed(tree.nodes):
        if isinstance(node.child, str):
            node_subset = selected_subset
            if node.condition_key:
                node_subset &= conditions[node.condition_key]
            value[node_subset] = variables[node.child][node_subset]
        else:
            _vec_when_tree_rec(
                tree=node.child,
                value=value,
                variables=variables,
                conditions=conditions,
                selected_subset=selected_subset & conditions[node.condition_key]
            )
            


def _when_tree_rec(
        tree: WhenTree,
        value: typing.Union[pd.Series, pd.DataFrame], 
        variables: typing.Union[pd.Series, pd.DataFrame], 
        conditions: pd.Series, 
        selected_subset: typing.Union[pd.Series, bool] = True
):
    raise NotImplemented("Coming soon to a store near you")

# def when_tree(
#     tree: WhenTree,
#     upcast_config: typing.Optional[dict] = None,
#     **kwargs: dict
# ):
#     upcast_config = upcast_config or {}
#     variables = {}
#     conditions = {}
#     for k, v in kwargs:
#         if k.startswith("var_"):
#             variables[k[4:]] = v
#         elif k.startswith("cond_"):
#             conditions[k[5:]] = v

def when_tree(
    tree: WhenTree,
    variables: dict,
    conditions: dict,
    upcast_config: typing.Optional[dict] = None,
) -> typing.Any:
    upcast_config = upcast_config or {}

    # TODO validate all leaf nodes are in tree
    assert len(variables) >= 1, "There must be at least one variable"
    assert len(conditions) >= 1, "There must be at least one condition"

    # Get the length over both variables and conditions
    target_len = -float('inf')
    for v in chain(variables.values(), conditions.values()):
        if isinstance(v, (pd.DataFrame,pd.Series)):
            target_len = max(len(v), target_len)
    if target_len == -float("inf"): target_len = None

    variables = helpers.upcast_variables(
        target_len=target_len, 
        **upcast_config, # Order matters, should be able to override len but not vars
        variables=variables, 
    )

    conditions = helpers.upcast_variables(
        variables=conditions, 
        target_columns=set(), # Ensure cast to nothing more than a series
        target_len=upcast_config.get("target_len", target_len) # Enable length override
    )

    default_value = None
    # for node in tree.nodes:
    if isinstance(tree.nodes[-1].child, str) and tree.nodes[-1].condition_key is None:
        # Root level default found
        default_value = variables[tree.nodes[-1].child]
        # break

    first_var = next(iter(variables.values()))
    if isinstance(first_var, (pd.DataFrame, pd.Series)):
        # Ensure the default_variable is cast to the correct type
        # This should already be true unless the value is none
        value: typing.Union[pd.DataFrame, pd.Series] = helpers.upcast_variables({
            "default": default_value, 
            "sample": first_var
        })["default"].copy()
        _vec_when_tree_rec(
            tree=tree,
            value=value,
            variables=variables,
            conditions=conditions,
        )
        return value