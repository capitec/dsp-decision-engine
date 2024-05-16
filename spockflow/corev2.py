from hamilton import node
from hamilton.function_modifiers import subdag
from hamilton.function_modifiers.recursive import assign_namespace
from hamilton.function_modifiers import base

from hamilton import node
from dataclasses import dataclass, field
import inspect
import typing



@dataclass
class VariableNodeFunctionParameters:
    """This is used to mark a function as a candidate to generate a node.

    Attributes:
        kwarg_input_generator (Callable): Is a function called with the parent class of the decorated function and a function name that outputs a dictionary of inputs_names -> datatypes to be used as kwargs to the function.

    """
    _FUNCTION_NODE_GENERATION_ATTR: typing.ClassVar = "_spock_variable_node_fn_params"

    upstream_internal_functions: typing.List[str] = field(default_factory=list)
    node_name_template: typing.Optional[str] = None
    is_nested_output: bool = True
    kwarg_input_generator: typing.Callable[[object,typing.Callable], typing.Dict[str,typing.Any]] = None
    
    def get_node_name(self, parent_name, function_name):
        node_name_template = self.node_name_template
        if node_name_template is None:
            node_name_template = "{function_name}" if self.is_nested_output else "{parent_name}"
        return node_name_template.format(
            parent_name=parent_name,
            function_name=function_name
        )
    
    def __call__(self, fn):
        setattr(
            fn,
            VariableNodeFunctionParameters._FUNCTION_NODE_GENERATION_ATTR,
            self
        )
        return fn
    
    def _replace_kwargs(self, n: node.Node, fn: typing.Callable) -> typing.Dict[str, typing.Any]:
        """This function takes in a node and returns arguments used to modify the node to replace the input parameters

        Args:
            n (node.Node): Node that this applies to
            fn (typing.Callable): the associated function

        Returns:
            typing.Dict[str, typing.Any]: a set of kwargs used in node.copy_with() to augment the input parameters.
        """
        if self.kwarg_input_generator is None: return {}
        if 'kwargs' not in n.input_types:
            raise ValueError(
                f"Could not find kwargs on function {fn}. "
                "When using a node with kwarg_input_generator the function must take in kwargs as a parameter."
            )
        # Call the input generator with the class defining the function and the function
        kw_arg_inputs = self.kwarg_input_generator(fn.__self__, fn)
        new_input_types = {**n.input_types}
        for k,v in kw_arg_inputs.items():
            if not isinstance(v, tuple): 
                # Automatically assume everything is required if not specified
                v = (v, node.DependencyType.REQUIRED)
            new_input_types[k] = v
        return {'input_types': new_input_types}
        
    
    def generate_node(self, fn: typing.Callable, parent_name: str, function_name: str, calling_module: "ModuleType"=None) -> node.Node:
        n = node.Node.from_fn(fn)
        copy_with_kwargs = {
            "name": self.get_node_name(parent_name=parent_name, function_name=function_name),
            **self._replace_kwargs(n ,fn)
        }
        if calling_module is not None:
            copy_with_kwargs["tags"] = {
                **n._tags,
                "module": calling_module.__name__
            }

        return n.copy_with(**copy_with_kwargs)


# This is a convenience alias to make the intention more clear to external users   
creates_node = VariableNodeFunctionParameters

class VariableNodeCreator(base.NodeCreator):
    """This is used to create variable nodes from a VariableNode class"""
    def generate_nodes(self, fn: "VariableNode", config: "Dict[str, Any]") -> "List[node.Node]":
        return fn._generate_nodes(fn.__name__, config)

    def validate(self, fn: "typing.Callable"):
        pass

class VariableNode:
    """This is a base class for all variable nodes to use"""
    def __init__(self,):
        # Others to consider
        # NodeExpander, NodeTransformer, NodeInjector, NodeDecorator (DefaultNodeDecorator())
        # Not really needed (core module does fine) COuld maybe use it for the new hamilton ui. TODO check what code looks like for varnodes
        # setattr(self, base.NodeResolver.get_lifecycle_name(), [SpockNodeResolver()])
        setattr(self, base.NodeCreator.get_lifecycle_name(), [VariableNodeCreator()])
        self._name = None
        self._module = None


    def _set_module(self, module: "ModuleType") -> typing.Self:
        """Called when this node is discovered with the name of the variable that defines it to set the name of this instance in the pipeline

        Args:
            name (str): Name of this instance

        Raises:
            ValueError: Raised when the same instance is declared with 2 names. E.g. item1 = Tree(), item2 = item1. It is unclear if a node should be created for item1 or item2 or an alias for both.

        Returns:
            VariableNode: This instance so that it can be used as part of a "builder" pattern.
        """
        if self._module is None:
            self._module = module
        if self._module is not module:
            raise ValueError(
                f"Cannot use the same node in different modules {self._module}.{self._name}!={self._module}.{self._name}.\n"
                f"If it is your intention to create an alias in the pipeline please use {self._name}={self._name}.alias().\n"
                f"If you would like to use the variable without including it in the pipeline please rename either \"{self._name}\"->\"_{name}\" or \"{self._name}\"->\"_{self._name}\"."
            )
        return self
    
    def _set_name(self, name: str) -> typing.Self:
        """Called when this node is discovered with the name of the variable that defines it to set the name of this instance in the pipeline

        Args:
            name (str): Name of this instance

        Raises:
            ValueError: Raised when the same instance is declared with 2 names. E.g. item1 = Tree(), item2 = item1. It is unclear if a node should be created for item1 or item2 or an alias for both.

        Returns:
            VariableNode: This instance so that it can be used as part of a "builder" pattern.
        """
        if self._name is None:
            self._name = name
        if self._name != name:
            raise ValueError(
                f"Cannot use the same node with different names {name}!={self._name}.\n"
                f"If it is your intention to create an alias in the pipeline please use {name}={self._name}.alias().\n"
                f"If you would like to use the variable without including it in the pipeline please rename either \"{name}\"->\"_{name}\" or \"{self._name}\"->\"_{self._name}\"."
            )
        return self
    
    @property
    def __name__(self):
        if self._name is None: 
            raise RuntimeError("Name of node used before it was set. This is likely a bug with how Spock generates nodes. Please report with a working example.")
        return self._name
    
    @staticmethod
    def _map_input_vars(n: node.Node, input_mapping: typing.Dict[str, str]) -> node.Node:
        """Maps namespaced variables to variables that can be accessed from a function.
        E.g. function1 under tree that makes use of a namespaced variable "tree.function2"
        can be defined as function1(function2: pd.DataFrame). This function does the mapping from
        "tree.function2" to "function2" before calling the function.

        Args:
            n (node.Node): The node that should be exposed.
            input_mapping (typing.Dict[str, str]): The mapping from source to dest inputs

        Returns:
            node.Node: Either the original node if no modifications were needed or a node that first maps the relevant columns and then calls the function.
        """

        should_replace = False
        new_input_types = {}
        namespaced_input_map = {}
        for key, value in n.input_types.items():
            if key in input_mapping: 
                should_replace = True
                mapped_key = input_mapping[key]
                new_input_types[mapped_key] = value
                namespaced_input_map[mapped_key] = key
            else:
                new_input_types[key] = value
                namespaced_input_map[key] = key
        if not should_replace: 
            return n
        
        current_fn = n.callable
        def new_function(**kwargs):
            kwargs_without_namespace = {
                namespaced_input_map[key]: value for key, value in kwargs.items()
            }
            # Have to translate it back to use the kwargs the fn is expecting
            return current_fn(**kwargs_without_namespace)

        return n.copy_with(input_types=new_input_types, callabl=new_function)
    
    @staticmethod
    def _does_define_node(item) -> bool:
        """Determines if a function is wrapped to create a node

        Args:
            item (typing.Callable): The member variable of the function

        Returns:
            bool: if the passed in item can create a node
        """
        return (
            isinstance(item, typing.Callable) and
            hasattr(item, VariableNodeFunctionParameters._FUNCTION_NODE_GENERATION_ATTR)
        )
    
    def _generate_nodes(self, name: str, config: "typing.Dict[str, typing.Any]") -> "typing.List[node.Node]":
        """Generate nodes for this class to be used in a hamilton dag

        Args:
            name (str): This name is used rather than self.__name__ to allow the same generate function to be used with multiple copies
            config (Dict[str, Any]): This is the hamilton config passed down to this class

        Returns:
            List[node.Node]: The resulting Hamilton nodes
        """
        node_functions = inspect.getmembers(self, predicate=self._does_define_node)
        namespaced_nodes = []
        nodes: typing.List[node.Node] = []
        node_input_mapping = {}
        for fn_name, node_fn in node_functions:
            var_node_params: VariableNodeFunctionParameters = getattr(
                node_fn, 
                VariableNodeFunctionParameters._FUNCTION_NODE_GENERATION_ATTR
            )
            nd = var_node_params.generate_node(node_fn, name, fn_name, self._module)
            if var_node_params.is_nested_output:
                namespaced_nodes.append(nd)
                node_input_mapping[fn_name] = assign_namespace(nd.name, name)
            else:
                nodes.append(nd)
                node_input_mapping[fn_name] = nd.name

        nodes += subdag.add_namespace(namespaced_nodes, name)
        nodes = [self._map_input_vars(n, node_input_mapping) for n in nodes]

        # Add a bit of early failure so that its easier to trace where the duplicate node originated from.
        node_names = set()
        for n in nodes:
            if n.name in node_names: 
                raise ValueError(f"Could not create {name} as it produced more than one node with the name {n.name}.")
            node_names.add(n.name)
        return nodes
    
    def alias(self) -> "AliasedVariableNode":
        """Creates a clone with a different name of all output nodes generated by this node.

        Returns:
            AliasedVariableNode: A variable node that generates outputs
        """
        if self._generate_nodes.__func__ is not VariableNode._generate_nodes:
            # Detected custom generate node class cannot be sure that the default aliasing method will work
            raise NotImplementedError("This function does not implement an aliasing method")
        return AliasedVariableNode(self)
    
    def clone(self) -> "CloneVariableNode":
        """Creates a complete copy of the subdag in which the all the features will be recomputed.
        NOTE: This isn't particularly useful. Please ensure you do not mean to use alias.

        Returns:
            CloneVariableNode: Generates a copy of all the nodes
        """
        return CloneVariableNode(self)
    
class AliasedVariableNode(VariableNode):
    def __init__(self, parent_node: VariableNode):
        super().__init__()
        self.parent_node = parent_node
    def _generate_nodes(self, name: str, config: "Dict[str, Any]") -> "List[node.Node]":
        # TODO this should output a list of functions that take in the outputs of one variable node and return them with a different name (do not redo compute)
        raise NotImplementedError()
    
    def alias(self) -> "AliasedVariableNode":
        # Reroute alias viability check to parent node
        return self.parent_node.alias()
    
class CloneVariableNode(VariableNode):
    def __init__(self, parent_node: VariableNode):
        super().__init__()
        self.parent_node = parent_node
    def _generate_nodes(self, name: str, config: "Dict[str, Any]") -> "List[node.Node]":
        return self.parent_node._generate_nodes(name, config)
    
    def alias(self) -> "AliasedVariableNode":
        # Reroute alias viability check to parent node
        return self.parent_node.alias()
    



class configure_output(subdag):
    @staticmethod
    def is_variable_node(v):
        return isinstance(v, VariableNode)
    
    @classmethod
    def get_variable_nodes(cls, m: "ModuleType"):
        var_nodes_w_names: typing.List[VariableNode] = inspect.getmembers(
            m, 
            predicate=cls.is_variable_node
        )
        return [
            var_node._set_name(name)._set_module(m) 
            for name,var_node in var_nodes_w_names
        ]

    def __init__(self, included_modules: "Union[ModuleType, Callable, VariableNode]", ignore_output=False):
        self.ignore_output=ignore_output
        if included_modules:
            self.do_generate=True
            super().__init__(*included_modules)

            for m in included_modules:
                # Only add modules
                if isinstance(m, typing.Callable): continue
                if isinstance(m, VariableNode): self.subdag_functions.append(m)
                self.subdag_functions += self.get_variable_nodes(m)
        else:
            self.config={}
            self.inputs={}
            self.subdag_functions=[]
            self.do_generate=True

    def generate_nodes(self, fn: "typing.Callable", configuration: "typing.Dict[str, typing.Any]") -> "typing.Collection[node.Node]":
        this_module = inspect.getmodule(fn)
        subdag_functions = self.subdag_functions+self.get_variable_nodes(this_module)
        # Resolve all nodes from passed in functions
        resolved_config = dict(configuration, **self.config)
        nodes = []
        for sd_fn in subdag_functions:
            for node_ in base.resolve_nodes(sd_fn, resolved_config):
                nodes.append(node_)
        # TODO might hook in here to add config
        # nodes += self._create_additional_static_nodes(nodes, namespace)
        # Add the final node that does the translation
        # nodes += [self.add_final_node(fn, final_node_name, namespace)]
        if not self.ignore_output:
            nodes.append(node.Node.from_fn(fn))
        return nodes


def initialize_spock_module(module_name, included_modules=None):
    import sys
    caller_module = sys.modules.get(module_name)
    if caller_module is None:
        raise ValueError(
            f"Could not initialise module with name {module_name}.\n"
            "Please ensure this function is called as initialize(__name__, ...).\n"
            "If you believe this to be an error please report it with details on the environment.\n"
            "Note that it is possible to use the following as a workaround:\n"
            "@configure_output(..., ignore_output=True)\n"
            "def initialise() -> None: pass\n"
        )
    @configure_output(included_modules, ignore_output=True)
    def spock_bootstrap_entrypoint_fn__() -> None:
        pass
    spock_bootstrap_entrypoint_fn__.__module__ = module_name
    setattr(caller_module, spock_bootstrap_entrypoint_fn__.__name__, spock_bootstrap_entrypoint_fn__)
