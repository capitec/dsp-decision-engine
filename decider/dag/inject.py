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
            
        Raises:
            ValueError: If any parameter mappings don't correspond to actual node parameters.
        """
        nodes_dict = self.expander.expand_nodes()
        
        # Start with all mappings as unused, remove as they get used
        unused_mappings = set(self.parameter_mapping.items())
        
        # Apply parameter mapping to each node
        remapped_nodes = {}
        for node_name, original_node in nodes_dict.items():
            remapped_node, used_mappings = self._map_input_vars(original_node, self.parameter_mapping)
            remapped_nodes[node_name] = remapped_node
            # Remove used mappings from unused set
            unused_mappings -= used_mappings
        
        # Check if any mappings were never used
        if unused_mappings:
            unused_dict = dict(unused_mappings)
            raise ValueError(
                f"Parameter mapping(s) not found in any node: {unused_dict}. "
                f"These internal parameter names don't exist in the expander's nodes."
            )
            
        return remapped_nodes
    
    @staticmethod
    def _map_input_vars(n: node.Node, input_mapping: t.Dict[str, str]) -> t.Tuple[node.Node, t.Set[t.Tuple[str, str]]]:
        """Maps external parameter names to internal parameter names for a node.
        
        Args:
            n: The node whose input parameters should be remapped
            input_mapping: Dictionary mapping external names to internal names
            
        Returns:
            A tuple of:
            - Either the original node if no remapping needed, or a new node with 
              remapped input parameters and a wrapper function
            - Set of (external_name, internal_name) tuples that were actually used
        """
        should_replace = False
        new_input_types = {}
        internal_to_external_map = {}
        used_mappings = set()
        
        # Check each input parameter of the node
        for internal_param, param_info in n.input_types.items():
            # Look for this internal parameter in our mapping (reverse lookup)
            external_param = None
            for ext_name, int_name in input_mapping.items():
                if int_name == internal_param:
                    external_param = ext_name
                    used_mappings.add((ext_name, int_name))
                    break
                    
            if external_param:
                should_replace = True
                new_input_types[external_param] = param_info
                internal_to_external_map[external_param] = internal_param
            else:
                new_input_types[internal_param] = param_info
                internal_to_external_map[internal_param] = internal_param
                
        if not should_replace:
            return n, used_mappings
            
        current_fn = n.callable
        
        def remapped_function(**kwargs):
            # Translate external parameter names back to internal names
            internal_kwargs = {
                internal_to_external_map[external_name]: value 
                for external_name, value in kwargs.items()
            }
            return current_fn(**internal_kwargs)
            
        return n.copy_with(input_types=new_input_types, callabl=remapped_function), used_mappings