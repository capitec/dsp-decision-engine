import typing
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from ..util import get_name, safe_update

NodeValue = typing.Union[pd.DataFrame, pd.Series, 'Tree', typing.Any]

class ConditionBuilder():
    def isin():
        pass
    def __ne__(self, __value: object) -> bool:
        pass
    def __eq__(self, __value: object) -> bool:
        pass

class InternalValue():
    pass

def Reject(code: int, description: str, soft_reject: bool=False) -> pd.DataFrame:
    df = pd.DataFrame(
        [dict(code = code, description=description, soft_reject=soft_reject)]
    )
    df._rule_engine_internal_prop_name_ = f"REJECT Code={code}"
    return df

@dataclass
class ConditionedNode:
    child: typing.Union['Tree',str]
    condition: typing.Optional[str] = None

@dataclass
class Tree:
    nodes: typing.List[ConditionedNode] = field(default_factory=list)
    is_vec: typing.Optional[bool] = None
    # TODO
    # value_types: typing.Dict[str, typing.Any] = field(default_factory=dict)
    # condition_types: typing.Dict[str, typing.Any] = field(default_factory=dict)
    def __post_init__(self):
        self._value_vars: typing.Optional[dict] = None
        self._condition_vars: typing.Optional[dict] = None
        self._is_root = True

    @property
    def value_vars(self) -> typing.Optional[dict]:
        if not self._is_root:
            # TODO po
            raise ValueError("Can only access the value_vars of the root node. This error can be present when using \"when\" or \"otherwise\" on a tree that has already been added to another tree.")
        if self._value_vars is None:
            self._value_vars = {}
        return self._value_vars

    @property
    def condition_vars(self) -> typing.Optional[dict]:
        if not self._is_root:
            raise ValueError("Can only access the condition_vars of the root node. This error can be present when using \"when\" or \"otherwise\" on a tree that has already been added to another tree.")
        if self._condition_vars is None:
            self._condition_vars = {}
        return self._condition_vars

    def when(
        self,
        condition: typing.Union[bool,pd.Series], 
        value: NodeValue,
        condition_name: typing.Optional[str] = None,
        value_name: typing.Optional[str] = None,
    ) -> 'Tree':
        if isinstance(value, Tree):
            safe_update(self.condition_vars, value.condition_vars)
            safe_update(self.value_vars, value.value_vars)
            value._is_root = False
            if self.is_vec is None and value.is_vec:
                self.is_vec = True
        else:
            value_name = get_name(value, value_name)
            safe_update(self.value_vars, {value_name: value})
            value = value_name
            
        if condition is not None:
            if self.is_vec is None and isinstance(condition, pd.Series):
                self.is_vec = True
            condition_name = get_name(condition, condition_name)
            safe_update(self.condition_vars, {condition_name: condition})
            condition = condition_name

        self.nodes.append(
            ConditionedNode(
                condition=condition, 
                child=value,
            ))
        return self
    
    def otherwise(
        self,
        value: NodeValue,
        value_name: typing.Optional[str] = None,
    ) -> 'Tree':
        return self.when(None, value, None, value_name)

    @classmethod
    def _vec_d_tree(
        cls,
        nodes: typing.List[ConditionedNode],
        variables: typing.Dict[str, typing.Union[pd.Series, pd.DataFrame]], 
        conditions: typing.Dict[str, typing.Union[pd.Series, bool]], 
    ) -> typing.Union[pd.DataFrame, pd.Series]:
        # convert all conditions into boolean series
        if "__default_value__" not in variables:
            variables["__default_value__"] = None

        normalised_conditions = {}
        vec_length = None
        for k,v in conditions.items():
            if isinstance(v, bool):
                v = pd.Series([v])
            elif isinstance(v, pd.Series):
                if vec_length is None or vec_length == 1:
                    vec_length = len(v)
                assert vec_length == len(v) or len(v) == 1, f"Cannot cast condition {k} with shape {len(v)} into length {vec_length}"
            else:
                raise ValueError(f"Condition {k}: Expected either a boolean or a pd.Series object but got {type(v)}")
            normalised_conditions[k] = v.astype(np.bool_)

        assert vec_length is not None, "Vector dtree must be called with at least one pd.Series condition"
        normalised_conditions = {
            k: v.repeat(vec_length) if len(v) == 1 else v
            for k,v in normalised_conditions.items()
        }

        # Convert all values to either Series or Dataframes
        df_cols = None
        for k,v in variables.items():
            if isinstance(v, pd.DataFrame):
                if df_cols is None:
                    df_cols = v.columns
                else:
                    assert set(v.columns) == set(df_cols), f"Expecting all dataframes to contain the same columns as inputs found {k} with columns {set(v.columns)} but expecting {set(df_cols)}."
            
            if isinstance(v, (pd.DataFrame, pd.Series)):
                assert len(v) == 1 or len(v) == vec_length, f"Found value {k} with length {len(v)} but expecting either length {vec_length} or 1"

        # Convert all variables to the correct type
        normalised_variables = {}
        for k, v in variables.items():
            if df_cols is None:
                if not isinstance(v, pd.Series):
                    v = pd.Series([v]*vec_length)
                elif len(v) == 1:
                    v = v.repeat(vec_length)
            else:
                if not isinstance(v, (pd.DataFrame, pd.Series)):
                    v = pd.Series([v]*vec_length)
                if isinstance(v, pd.Series):
                    if len(v) == 1:
                        v = v.repeat(vec_length)
                    if v.name in df_cols:
                        v = pd.DataFrame({k: v if k == v.name else [None]*vec_length for k in df_cols})
                    else:
                        v = pd.DataFrame({k: v for k in df_cols})
                elif isinstance(v, pd.DataFrame):
                    if len(v) == 1:
                        v = pd.concat([v]*vec_length, ignore_index=True)
                    
            normalised_variables[k] = v

        # ensure default value has the right dtype
        default_value = normalised_variables["__default_value__"]
        try:
            ref_key = next((k for k in normalised_variables.keys() if k != "__default_value__"))
            ref_val: typing.Union[pd.Series, pd.DataFrame] = normalised_variables[ref_key]
            contains_na = default_value.isna().any()
            if isinstance(ref_val, pd.Series):
                target_types = ref_val.dtype
                # NA values cannot be stored in an integer type
                if contains_na and pd.api.types.is_integer_dtype(target_types):
                    target_types = np.float64
            elif isinstance(ref_val, pd.DataFrame):
                target_types = ref_val.dtypes
                for type_k, type_v in target_types.items():
                    if contains_na[type_k] and pd.api.types.is_integer_dtype(type_v):
                        target_types[type_k] = np.float64
            default_value = default_value.astype(target_types)
            # if isinstance(ref_val, pd.Series):
            # elif isinstance(ref_val, pd.DataFrame):
            #     normalised_variables["__default_value__"] = normalised_variables["__default_value__"].convert_dtypes(ref_val.dtypes)

        except StopIteration:
            pass
        normalised_variables["__default_value__"] = default_value
        res = default_value.copy()
        cls._vec_when_tree_rec(
            nodes=nodes,
            value=res,
            variables=normalised_variables,
            conditions=normalised_conditions,
        )
        return res
    
    @classmethod
    def _vec_when_tree_rec(
        cls,
        nodes: typing.List[ConditionedNode],
        value: typing.Union[pd.Series, pd.DataFrame], 
        variables: typing.Dict[str,typing.Union[pd.Series, pd.DataFrame]], 
        conditions: typing.Dict[str,pd.Series], 
        selected_subset: typing.Optional[pd.Series] = None
    ):
        # Done in reverse as earliest nodes take presence
        if selected_subset is None:
            selected_subset = pd.Series([True]*len(value))
        for node in reversed(nodes):
            if isinstance(node.child, str):
                node_subset = selected_subset.copy()
                if node.condition:
                    node_subset &= conditions[node.condition]
                # Values will ignore index but is also less safe if the indicies do not align
                value[node_subset.values] = variables[node.child][node_subset.values]
            else:
                cls._vec_when_tree_rec(
                    nodes=node.child.nodes,
                    value=value,
                    variables=variables,
                    conditions=conditions,
                    selected_subset= selected_subset if node.condition is None else (selected_subset & conditions[node.condition])
                )

    @classmethod
    def _scalar_d_tree(
        cls,
        nodes: typing.List[ConditionedNode],
        variables: typing.Dict[str, typing.Union[pd.Series, pd.DataFrame]], 
        conditions: typing.Dict[str, bool], 
    ) -> typing.Union[pd.DataFrame, pd.Series]:
        # Do DFS for first root node
        nodes_to_explore = list(reversed(nodes)) # ensure nodes are visited in add order
        while nodes_to_explore:
            # Remove from end and adding to end FIFO = DFS
            node = nodes_to_explore.pop()
            if node.condition is None or conditions[node.condition]:
                if isinstance(node.child, str): return variables[node.child]
                nodes_to_explore.extend(reversed(node.child.nodes))
        if "__default_value__" in variables: return variables["__default_value__"]
        return None # TODO consider throwing an error

    def execute(self, values=None, conditions=None):
        values = values or {}
        conditions = conditions or {}
        values = {**self.value_vars, **values}
        conditions = {**self.condition_vars, **conditions}
        if self.is_vec is None:
            for v in conditions.values():
                if isinstance(v, pd.Series):
                    self.is_vec = True
                    break
        if self.is_vec:
            return self._vec_d_tree(self.nodes, values, conditions)
        return self._scalar_d_tree(self.nodes, values, conditions)
    
    def visualize(self):
        pass
