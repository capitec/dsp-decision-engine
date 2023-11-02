import typing
import pandas as pd
from hamilton import function_modifiers as fm

class OverrideNodeExpander:
    def __init__(self, fn, node_expander):
        self.node_expander = node_expander
        if node_expander is None:
            return 
        lifecycle_name = node_expander.get_lifecycle_name()
        if hasattr(fn, node_expander.get_lifecycle_name()):
            raise ValueError("Steps can only contain input mappings if no other node expanders exist")
        if not getattr(fn, "_skip_type_validation_", False):
            node_expander.validate(fn)
        self.fn = fn
        self.lifecycle_name=lifecycle_name

    def __enter__(self):
        if self.node_expander is None: return
        setattr(self.fn, self.lifecycle_name, [self.node_expander])

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.node_expander is None: return
        delattr(self.fn, self.lifecycle_name)


def infer_inject_parameter(input_value: typing.Union[str, dict], raise_on_fail=True):
    if isinstance(input_value, str):
        return fm.source(input_value)
    if isinstance(input_value, dict):
        if "row" in input_value:
            additional_kwargs = {}
            if all((not isinstance(v, list) for v in input_value["row"].values())):
                additional_kwargs["index"] = [0]
            return fm.value(pd.DataFrame(input_value["row"], **additional_kwargs))
        if "value" in input_value:
            return fm.value(input_value["value"])
        if "source" in input_value:
            return fm.source(input_value["source"])
    if raise_on_fail:
        raise ValueError("input must be either a string or a dict containing a value or a source attribute")


def upcast_variables(variables: dict, target_columns = None, target_len = None, fill_value=None) -> dict:
    # Determine the number of rows and columns
    _target_columns = set()
    _target_len = -float('inf')

    for v in variables.values():
        if target_len is None and isinstance(v, (pd.DataFrame,pd.Series)):
            _target_len = max(len(v), _target_len)
        if target_columns is None and isinstance(v, pd.DataFrame):
            _target_columns.update(v.columns)
    
    if target_columns is None: target_columns = _target_columns
    if target_len is None: target_len = _target_len 
    

    if target_len == -float('inf') and len(target_columns) == 0:
        # All variables are primitive
        return variables
    
    out = {}
    # Upcast all to at least a series and ensure length
    for k, v in variables.items():
        if not isinstance(v, (pd.DataFrame,pd.Series)):
            v = pd.Series([v]*target_len)
        if len(v) == 1:
            if isinstance(v, pd.Series):
                v = v.repeat(target_len)
            else:
                v = pd.concat([v]*target_len, ignore_index=True)
        elif len(v) != target_len:
            raise ValueError(f"Cannot cast variable {k} with length {len(v)} into length {target_len}")
        out[k] = v
    
    if len(target_columns) == 0:
        # Cast all to series type
        for k, v in out.items():
            # Downcast DataFrames
            if isinstance(v, pd.DataFrame):
                if len(v.columns) != 1:
                    raise "Cannot cast a DataFrame with more than one column into a pd.Series"
                else:
                    out[k] = v[v.columns[0]]
        return out
    
    # Cast everything else to a DataFrames
    target_columns_list = list(target_columns)
    for k, v in out.items():
        if isinstance(v, pd.Series):
            v = pd.DataFrame({k: v for k in target_columns_list})
        if isinstance(v, pd.DataFrame):
            difference_set = target_columns - set(v.columns)
            if len(difference_set) > 0:
                v[list(difference_set)] = fill_value
            v = v[target_columns_list] # Drop any extra columns and ensure ordering
        out[k] = v
    return out