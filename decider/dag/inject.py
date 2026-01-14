import typing as t
from dataclasses import dataclass
from hamilton import node
from .core import DeciderExpandableModule

if t.TYPE_CHECKING:
    from hamilton import node

@dataclass
class InjectedModule(DeciderExpandableModule):
    """A module wrapper that remaps input parameters for nodes from an inner expander.
    
    This allows connecting external parameters to internal parameter names by remapping
    the node dependencies. For example, mapping "namespace.value1" to "internal_param1".
    
    Attributes:
        expander: The inner expandable module whose nodes will have parameters remapped
        parameter_mapping: Dictionary mapping external parameter names to internal names
                          e.g., {"namespace.value1": "internal_param1"}
    """
    expander: DeciderExpandableModule
    parameter_mapping: t.Dict[str, str]
    
    def expand_nodes(self) -> t.Dict[str, "node.Node"]:
        """Expands nodes from the inner expander and remaps their input parameters.
        
        Returns:
            A dictionary mapping node names to nodes with remapped input parameters.
        """
        nodes_dict = self.expander.expand_nodes()
        
        # Apply parameter mapping to each node
        remapped_nodes = {}
        for node_name, original_node in nodes_dict.items():
            remapped_node = self._map_input_vars(original_node, self.parameter_mapping)
            remapped_nodes[node_name] = remapped_node
            
        return remapped_nodes
    
    @staticmethod
    def _map_input_vars(n: node.Node, input_mapping: t.Dict[str, str]) -> node.Node:
        """Maps external parameter names to internal parameter names for a node.
        
        Args:
            n: The node whose input parameters should be remapped
            input_mapping: Dictionary mapping external names to internal names
            
        Returns:
            Either the original node if no remapping needed, or a new node with 
            remapped input parameters and a wrapper function.
        """
        should_replace = False
        new_input_types = {}
        internal_to_external_map = {}
        
        # Check each input parameter of the node
        for internal_param, param_info in n.input_types.items():
            # Look for this internal parameter in our mapping (reverse lookup)
            external_param = None
            for ext_name, int_name in input_mapping.items():
                if int_name == internal_param:
                    external_param = ext_name
                    break
                    
            if external_param:
                should_replace = True
                new_input_types[external_param] = param_info
                internal_to_external_map[external_param] = internal_param
            else:
                new_input_types[internal_param] = param_info
                internal_to_external_map[internal_param] = internal_param
                
        if not should_replace:
            return n
            
        current_fn = n.callable
        
        def remapped_function(**kwargs):
            # Translate external parameter names back to internal names
            internal_kwargs = {
                internal_to_external_map[external_name]: value 
                for external_name, value in kwargs.items()
            }
            return current_fn(**internal_kwargs)
            
        return n.copy_with(input_types=new_input_types, callabl=remapped_function)