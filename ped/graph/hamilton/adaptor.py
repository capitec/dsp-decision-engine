import typing as t
import inspect
from types import ModuleType
from dataclasses import dataclass, field
from hamilton.lifecycle.base import BasePostGraphConstruct
from hamilton import node as hamilton_node
from ped.modules import ConstructedGraphModules, GraphModule, PEDNode

if t.TYPE_CHECKING:
    from hamilton import graph


def wrap_function_inputs(
    func: t.Callable, 
    input_mapping: t.Dict[str, str], 
    static_kwargs: t.Dict[str, t.Any] = None
) -> t.Callable:
    """Wraps a function to remap its input parameters and inject static kwargs.
    
    Args:
        func: The original function to wrap
        input_mapping: A dictionary mapping internal function parameters to external variable names
                       e.g., {"internal_param": "external_var"} maps function's "internal_param" to external "external_var"
        static_kwargs: Static keyword arguments to inject into the function (e.g., config objects)
    
    Returns:
        A wrapped function that accepts the external parameter names and internally maps them to the original function's parameters.
    """
    # Pre-compute the reverse mapping at graph build time (outside the wrapper)
    external_to_internal = {external_var: internal_param for internal_param, external_var in input_mapping.items()}
    static_kwargs = static_kwargs or {}
    
    def wrapper(**kwargs):
        # Use pre-computed mapping for fast execution
        mapped_kwargs = {}
        for external_var, value in kwargs.items():
            internal_param = external_to_internal.get(external_var, external_var)
            mapped_kwargs[internal_param] = value
        
        # Merge in static kwargs (static kwargs take precedence)
        final_kwargs = {**mapped_kwargs, **static_kwargs}
        return func(**final_kwargs)
    
    return wrapper


@dataclass
class DeciderAdaptorHook(BasePostGraphConstruct):
    modules: ConstructedGraphModules = field(default_factory=ConstructedGraphModules)

    def add_module(self, mod: GraphModule):
        self.modules.root.append(mod)

    def post_graph_construct(
        self,
        *,
        graph: "graph.FunctionGraph",
        modules: ConstructedGraphModules,
        config: t.Dict[str, t.Any],
    ):
        extra_nodes: t.Dict[str, "hamilton_node.Node"] = {}
        
        # Convert PED modules to Hamilton nodes
        for module in self.modules.root:
            ped_nodes = module.root.module_namespaced_nodes(module.root.name)
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
        # Wrap the callable with input mapping and static kwargs
        wrapped_callable = wrap_function_inputs(
            ped_node.callable, 
            ped_node.input_map, 
            ped_node.static_kwargs
        )
        
        # Create Hamilton node with original function to get proper typing
        original_node = hamilton_node.Node.from_fn(ped_node.original_callable)
        

        # Build input_types for Hamilton node from the external variable names
        # Exclude static_kwargs parameters as they don't come from the graph
        hamilton_input_types = {}
        for internal_param, external_var in ped_node.input_map.items():
            # Skip parameters that are provided by static_kwargs
            if internal_param not in ped_node.static_kwargs:
                if internal_param in original_node.input_types:
                    hamilton_input_types[external_var] = original_node.input_types[internal_param]
                else:
                    # Fallback for cases where parameter info isn't available
                    hamilton_input_types[external_var] = (t.Any, hamilton_node.DependencyType.REQUIRED)
        
        # Create Hamilton node
        hamilton_nodes[ped_node.namespaced_name] = original_node.copy_with(
            name=ped_node.name,
            typ=original_node.type,
            callabl=wrapped_callable,
            input_types=hamilton_input_types,
            namespace=ped_node.namespace,
            tags={**getattr(original_node, 'tags', {}), **ped_node.extra},
        )
    
    return hamilton_nodes
