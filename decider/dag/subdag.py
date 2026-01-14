import typing as t
from dataclasses import dataclass
from hamilton.function_modifiers import subdag
from .core import DeciderExpandableModule

if t.TYPE_CHECKING:
    from hamilton import node

@dataclass
class NamespacedModule(DeciderExpandableModule):
    """A module wrapper that adds a namespace prefix to all nodes from an inner expander.
    
    This allows reusing the same expandable module multiple times in a DAG without
    node name conflicts by prefixing all node names with the specified namespace.
    
    Attributes:
        namespace: The namespace prefix to add to all node names
        expander: The inner expandable module whose nodes will be namespaced
    """
    namespace: str
    expander: DeciderExpandableModule
    
    def expand_nodes(self) -> t.Dict[str, "node.Node"]:
        """Expands nodes from the inner expander and adds namespace prefixes.
        
        Returns:
            A dictionary mapping namespaced node names to their corresponding nodes.
            Each node name will be prefixed with the namespace (e.g., 'namespace.node_name').
        """
        nodes_dict = self.expander.expand_nodes()
        nodes_list = list(nodes_dict.values())
        
        namespaced_nodes = subdag.add_namespace(nodes_list, self.namespace)
        
        return {node.name: node for node in namespaced_nodes}