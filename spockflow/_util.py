
import typing

if typing.TYPE_CHECKING:
    import types
    from spockflow.nodes import VariableNodeExpander


def is_variable_node(fn):
    from spockflow.nodes import VariableNodeExpander
    return isinstance(fn, VariableNodeExpander)

def get_variable_nodes(modules: typing.List["types.ModuleType"]) -> typing.List["VariableNodeExpander"]:
    import inspect
    
    variable_nodes = []
    for m in modules:
        for v_node_name, v_node in inspect.getmembers(m, predicate=is_variable_node):
            v_node.set_name(v_node_name)
            v_node.set_module(m.__name__)
            variable_nodes.append(v_node)
    # Second loop through m to ensure these are only added after module defined nodes
    for m in modules:
        for out_idx, v_node in enumerate(getattr(m, "OUTPUTS", [])):
            if not is_variable_node(v_node): continue
            try:
                v_node_name = v_node.name
            except ValueError as e:
                raise ValueError(f"Output {out_idx} of module {m.__name__} is not declared as a variable first. Please explicitly set \"var = node\" in code before adding the node as an output") from e
            v_node.set_module(m.__name__)
            if v_node not in variable_nodes:
                variable_nodes.append(v_node)
    return variable_nodes

def init_module_variable_node_names(mods: typing.List["types.ModuleType"]):
    get_variable_nodes(mods)

def get_name(value, value_name: typing.Optional[str]):
    import types
    import pandas as pd

    if value_name is not None: return value_name
    if isinstance(value, pd.Series) and value.name is not None: return value.name
    if isinstance(value, pd.DataFrame) and value.attrs.get("name") is not None: return value.attrs.get("name")
    if isinstance(value, types.FunctionType): return value.__name__
    internal_name = getattr(value, "_rule_engine_internal_prop_name_", None)
    if internal_name is not None: return internal_name
    internal_name = getattr(value, "name", None)
    if internal_name is not None: return internal_name
    raise ValueError(f"Could not infer property name. Please manually provide a name for the value {value}")

def safe_update(dict_new: dict, dict_old: dict):
    for k, v in dict_old.items():
        if k in dict_new:
            assert dict_new[k] is dict_old[k], f"Value {k} used multiple times with different input values"
        else:
            dict_new[k] = v

def get_direct_modules(mod: "types.ModuleType") -> typing.List["types.ModuleType"]:
    direct_modules = getattr(mod, "DIRECT_MODULES", [])
    out = list(direct_modules)
    for m in direct_modules:
        out.extend(get_direct_modules(m))
    return out

def get_mod_outputs(mod: "types.ModuleType") -> typing.List[typing.Union[str, typing.Callable, "VariableNodeExpander"]]:
    import types
    from spockflow.nodes import VariableNodeExpander

    outputs = getattr(mod, "OUTPUTS", None)
    assert isinstance(outputs, list), f"Expecting global variable OUTPUTS to be defined on module {mod.__name__}"
    out = []
    for o in outputs:
        if isinstance(o, types.ModuleType):
            out.extend(get_mod_outputs(o))
        elif isinstance(o, VariableNodeExpander):
            out.append(o)
        elif isinstance(o, (str, typing.Callable)) or hasattr(o, "name"):
            out.append(o)
        else:
            raise (ValueError(f"Invalid output defined {o} in module {mod.__name__}. Expecting a function, module, string or an element with a name property"))
    return out

def get_output_types(outputs: typing.List[typing.Union["VariableNodeExpander", str, typing.Callable]]):
    import sys
    from spockflow.nodes import VariableNodeExpander

    out_typ = []
    for output in outputs:
        if isinstance(output, typing.Callable):
            # See hamilton.nodes.Node.from_fn
            type_hint_kwargs = {} if sys.version_info < (3, 9) else {"include_extras": True}
            return_type = typing.get_type_hints(output, **type_hint_kwargs).get("return")
            if return_type is None:
                raise ValueError(f"Missing type hint for return value in function {output.__qualname__}.")
            out_typ.append(return_type)
        elif isinstance(output, VariableNodeExpander):
            out_typ.append(output.get_return_type())
        else:
            out_typ.append(typing.Any)
    return out_typ
