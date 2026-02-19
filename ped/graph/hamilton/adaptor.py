import typing as t
from types import ModuleType
from dataclasses import dataclass, field
from hamilton.lifecycle.base import BasePostGraphConstruct
from hamilton import node as hamilton_node
from ped.modules import ConstructedGraphModules, GraphModule, PEDNode

if t.TYPE_CHECKING:
    from hamilton import graph


def wrap_function_inputs(func: t.Callable, input_mapping: t.Dict[str, str]) -> t.Callable:
    """Wraps a function to remap its input parameters based on the provided mapping.
    
    Args:
        func: The original function to wrap
        input_mapping: A dictionary mapping internal function parameters to external variable names
                       e.g., {"internal_param": "external_var"} maps function's "internal_param" to external "external_var"
    
    Returns:
        A wrapped function that accepts the external parameter names and internally maps them to the original function's parameters.
    """
    # Pre-compute the reverse mapping at graph build time (outside the wrapper)
    external_to_internal = {external_var: internal_param for internal_param, external_var in input_mapping.items()}
    
    def wrapper(**kwargs):
        # Use pre-computed mapping for fast execution
        mapped_kwargs = {}
        for external_var, value in kwargs.items():
            internal_param = external_to_internal.get(external_var, external_var)
            mapped_kwargs[internal_param] = value
        return func(**mapped_kwargs)
    
    return wrapper


@dataclass
class DeciderAdaptorHook(BasePostGraphConstruct):
    modules: ConstructedGraphModules = field(default_factory=ConstructedGraphModules)

    def add_module(self, key: str, mod: GraphModule):
        self.modules.root[key] = mod


    def post_graph_construct(
        self,
        *,
        graph: "graph.FunctionGraph",
        modules: t.List[ModuleType],
        config: t.Dict[str, t.Any],
    ):
        extra_nodes: t.Dict[str, "hamilton_node.Node"] = {}
        
        # Convert PED modules to Hamilton nodes
        for module_name, module in self.modules.root.items():
            ped_nodes = module.module_namespaced_nodes(module_name)
            hamilton_nodes = ped_nodes_to_hamilton_nodes(ped_nodes)
            extra_nodes.update(hamilton_nodes)
        
        graph.nodes = graph.with_nodes(extra_nodes).nodes


def ped_nodes_to_hamilton_nodes(ped_nodes: t.List["PEDNode"]) -> t.Dict[str, "hamilton_node.Node"]:
    """
    Convert PED nodes to Hamilton nodes.
    
    Args:
        ped_nodes: List of PED nodes with namespace and input mapping already applied
        
    Returns:
        Dictionary mapping fully qualified node names to Hamilton nodes
    """
    
    hamilton_nodes = {}
    for ped_node in ped_nodes:
        # Wrap the callable with input mapping
        wrapped_callable = wrap_function_inputs(ped_node.callable, ped_node.input_map)
        
        # Create Hamilton node with original function to get proper typing
        original_node = hamilton_node.Node.from_fn(ped_node.original_callable)
        
        # Build input_types for Hamilton node from the external variable names
        hamilton_input_types = {}
        for internal_param, external_var in ped_node.input_map.items():
            if internal_param in original_node.input_types:
                hamilton_input_types[external_var] = original_node.input_types[internal_param]
            else:
                # Fallback for cases where parameter info isn't available
                hamilton_input_types[external_var] = (t.Any, hamilton_node.DependencyType.REQUIRED)
        
        # Create Hamilton node
        hamilton_nodes[ped_node.namespaced_name] = original_node.copy_with(
            name=ped_node.namespaced_name,
            typ=original_node.typ,
            callabl=wrapped_callable,
            input_types=hamilton_input_types,
            namespace=ped_node.namespace,
        )
    
    return hamilton_nodes
