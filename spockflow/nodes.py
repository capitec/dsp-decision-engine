import typing
import inspect
from typing_extensions import Self
from pydantic import BaseModel
from dataclasses import dataclass, field

from hamilton import node
from hamilton.function_modifiers import subdag
from hamilton.function_modifiers.recursive import assign_namespace
from hamilton.function_modifiers import base

if typing.TYPE_CHECKING:
    from types import ModuleType
    from spockflow.config_node import ConfigVariableNode, ConfigCacheStrategy
    from hamilton import driver


@dataclass
class VariableNodeFunctionParameters:
    """This is used to mark a variable node function as a candidate to generate a node.

    The following example makes use of the creates_node alias to this class.
    class ExampleNode(VariableNode):
        @creates_node()
        def internal_function(self, test: int) -> int:
            return test

        @creates_node(is_namespaced=False)
        def exposed_function(self, internal_function: int) -> int:
            return internal_function

    Attributes:
        kwarg_input_generator (Callable): Is a function called with the parent class of the decorated function and a function name that outputs a dictionary of inputs_names -> datatypes to be used as kwargs to the function.

    """

    _FUNCTION_NODE_GENERATION_ATTR: typing.ClassVar = "_spock_variable_node_fn_params"

    upstream_internal_functions: typing.List[str] = field(default_factory=list)
    node_name_template: typing.Optional[str] = None
    is_namespaced: bool = True
    kwarg_input_generator: typing.Union[
        typing.Callable[[object, typing.Callable], typing.Dict[str, typing.Any]], str
    ] = None

    def get_node_name(self, parent_name, function_name):
        node_name_template = self.node_name_template
        if node_name_template is None:
            node_name_template = (
                "{function_name}" if self.is_namespaced else "{parent_name}"
            )
        return node_name_template.format(
            parent_name=parent_name, function_name=function_name
        )

    def __call__(self, fn):
        setattr(fn, VariableNodeFunctionParameters._FUNCTION_NODE_GENERATION_ATTR, self)
        return fn

    def _replace_kwargs(
        self, n: node.Node, fn: typing.Callable
    ) -> typing.Dict[str, typing.Any]:
        """This function takes in a node and returns arguments used to modify the node to replace the input parameters

        Args:
            n (node.Node): Node that this applies to
            fn (typing.Callable): the associated function

        Returns:
            typing.Dict[str, typing.Any]: a set of kwargs used in node.copy_with() to augment the input parameters.
        """
        if self.kwarg_input_generator is None:
            return {}
        if "kwargs" not in n.input_types:
            raise ValueError(
                f"Could not find kwargs on function {fn}. "
                "When using a node with kwarg_input_generator the function must take in kwargs as a parameter."
            )

        if isinstance(self.kwarg_input_generator, str):
            # Call the bound method of the class defining the function with the function
            kw_arg_inputs = getattr(fn.__self__, self.kwarg_input_generator)(fn)
        else:
            # Call the input generator with the class defining the function and the function
            kw_arg_inputs = self.kwarg_input_generator(fn.__self__, fn)
        new_input_types = {**n.input_types}
        new_input_types.pop("kwargs", None)
        for k, v in kw_arg_inputs.items():
            if not isinstance(v, tuple):
                # Automatically assume everything is required if not specified
                v = (v, node.DependencyType.REQUIRED)
            new_input_types[k] = v
        return {"input_types": new_input_types}

    def generate_node(
        self,
        fn: typing.Callable,
        parent_name: str,
        function_name: str,
        calling_module: "ModuleType" = None,
    ) -> node.Node:
        n = node.Node.from_fn(fn)
        copy_with_kwargs = {
            "name": self.get_node_name(
                parent_name=parent_name, function_name=function_name
            ),
            **self._replace_kwargs(n, fn),
        }
        if calling_module is not None:
            copy_with_kwargs["tags"] = {**n._tags, "module": calling_module.__name__}

        return n.copy_with(**copy_with_kwargs)


# This is a convenience alias to make the intention more clear to external users
creates_node = VariableNodeFunctionParameters


class VariableNodeCreator(base.NodeCreator):
    """This is used to create variable nodes from a VariableNode class"""

    def generate_nodes(
        self, fn: "VariableNode", config: "typing.Dict[str, typing.Any]"
    ) -> "typing.List[node.Node]":
        return fn._generate_nodes(fn.__name__, config)

    def validate(self, fn: "typing.Callable"):
        pass

    def required_config(self) -> typing.Optional[typing.List[str]]:
        """Override this to None to disable filtering"""
        return None


assert (
    base.NodeCreator.get_lifecycle_name() == "generate"
), "The version of Hamilton you are using is incompatible with this generation"


class VariableNode(BaseModel):
    """This is a base class for all variable nodes to use"""

    _name: typing.Optional[str] = None
    _module: typing.Optional[str] = None
    # NOTE: this assumes that base.NodeCreator.get_lifecycle_name() returns generate
    generate: typing.ClassVar[typing.List[base.NodeCreator]] = [VariableNodeCreator()]

    # def model_post_init(self, __context):
    #     setattr(self, base.NodeCreator.get_lifecycle_name(), [VariableNodeCreator()])

    def _set_module(self, module: "ModuleType") -> "Self":
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
                f'If you would like to use the variable without including it in the pipeline please rename either "{self._name}"->"_{name}" or "{self._name}"->"_{self._name}".'
            )
        return self

    def _set_name(self, name: str) -> "Self":
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
                f'If you would like to use the variable without including it in the pipeline please rename either "{name}"->"_{name}" or "{self._name}"->"_{self._name}".'
            )
        return self

    @property
    def __name__(self):
        # Tough removing this check but it hinders the creating of nodes as the inspect.get_instances triggeres this property invokation often before it is set.
        # if self._name is None:
        #     return None
        # raise RuntimeError("Name of node used before it was set. This is likely a bug with how Spock generates nodes. Please report with a working example.")
        return self._name

    @staticmethod
    def _map_input_vars(
        n: node.Node, input_mapping: typing.Dict[str, str]
    ) -> node.Node:
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
        return callable(item) and hasattr(
            item, VariableNodeFunctionParameters._FUNCTION_NODE_GENERATION_ATTR
        )

    def compile(self):
        """
        This is a function that enables a variable node to finalize configuration before producing nodes.
        """
        return self

    def _generate_runtime_nodes(
        self, config: "typing.Dict[str, typing.Any]", compiled_node: "Self"
    ) -> "typing.List[node.Node]":
        """Generate nodes that are needed when executing this node as a standalone function outside of the context of a Hamilton Dag

        Args:
            config (typing.Dict[str, typing.Any]): Additional Configuration

        Returns:
            typing.List[node.Node]: The Generated nodes
        """
        return []

    def _generate_nodes(
        self,
        name: str,
        config: "typing.Dict[str, typing.Any]",
        include_runtime_nodes: bool = False,
    ) -> "typing.List[node.Node]":
        """Generate nodes for this class to be used in a hamilton dag

        Args:
            name (str): This name is used rather than self.__name__ to allow the same generate function to be used with multiple copies
            config (Dict[str, Any]): This is the hamilton config passed down to this class
            include_runtime_nodes (bool): Used to include additional nodes needed during runtime execution

        Returns:
            List[node.Node]: The resulting Hamilton nodes
        """
        compiled_variable_node = self.compile()
        node_functions = inspect.getmembers(
            compiled_variable_node, predicate=self._does_define_node
        )
        namespaced_nodes = []
        nodes: typing.List[node.Node] = []
        node_input_mapping = {}
        for fn_name, node_fn in node_functions:
            var_node_params: VariableNodeFunctionParameters = getattr(
                node_fn, VariableNodeFunctionParameters._FUNCTION_NODE_GENERATION_ATTR
            )
            nd = var_node_params.generate_node(node_fn, name, fn_name, self._module)
            if var_node_params.is_namespaced:
                namespaced_nodes.append(nd)
                node_input_mapping[fn_name] = assign_namespace(nd.name, name)
            else:
                nodes.append(nd)
                node_input_mapping[fn_name] = nd.name

        nodes += subdag.add_namespace(namespaced_nodes, name)
        nodes = [self._map_input_vars(n, node_input_mapping) for n in nodes]
        if include_runtime_nodes:
            nodes.extend(self._generate_runtime_nodes(config, compiled_variable_node))

        # Add a bit of early failure so that its easier to trace where the duplicate node originated from.
        node_names = set()
        for n in nodes:
            if n.name in node_names:
                raise ValueError(
                    f"Could not create {name} as it produced more than one node with the name {n.name}."
                )
            node_names.add(n.name)
        return nodes

    def alias(self) -> "AliasedVariableNode":
        """Creates a clone with a different name of all output nodes generated by this node.

        Returns:
            AliasedVariableNode: A variable node that generates outputs
        """
        if self._generate_nodes.__func__ is not VariableNode._generate_nodes:
            # Detected custom generate node class cannot be sure that the default aliasing method will work
            raise NotImplementedError(
                "This function does not implement an aliasing method"
            )
        return AliasedVariableNode(self)

    def clone(self) -> "CloneVariableNode":
        """Creates a complete copy of the subdag in which the all the features will be recomputed.
        NOTE: This isn't particularly useful. Please ensure you do not mean to use alias.

        Returns:
            CloneVariableNode: Generates a copy of all the nodes
        """
        return CloneVariableNode(self)

    def get_driver(
        self,
        config: "Dict[str, Any]",
        name: str = None,
        adapter: "Optional[Union[lifecycle_base.LifecycleAdapter, List[lifecycle_base.LifecycleAdapter]]]" = None,
        _graph_executor: "GraphExecutor" = None,
        _use_legacy_adapter: bool = True,
    ) -> "driver.Driver":
        from hamilton import driver
        from hamilton import graph

        dr = driver.Driver(
            config=config,
            adapter=adapter,
            _graph_executor=_graph_executor,
            _use_legacy_adapter=_use_legacy_adapter,
        )
        if name is None:
            name = self.__class__.__name__
        fg = graph.FunctionGraph(
            nodes={
                n.name: n
                for n in self._generate_nodes(name, config, include_runtime_nodes=True)
            },
            config=config,
            adapter=adapter,
        )
        # Need to run this to create links between nodes
        dr.graph = graph.FunctionGraph(
            config=config,
            adapter=adapter,
            nodes=graph.create_function_graph(config=config, adapter=adapter, fg=fg),
        )
        return dr

    def execute(self, inputs, config=None, final_vars=None, overrides=None, name=None):
        if name is None:
            name = self.__class__.__name__
        if config is None:
            config = {}
        dr = self.get_driver(config=config, name=name)
        if final_vars is None:
            final_vars = [name]
        return dr.execute(final_vars=final_vars, overrides=overrides, inputs=inputs)

    @classmethod
    def from_config(
        cls,
        config_path: str = "",
        caching_strategy: typing.Optional["ConfigCacheStrategy"] = None,
    ) -> "ConfigVariableNode[Self]":
        from spockflow.config_node import ConfigVariableNode

        kwargs = dict()
        if caching_strategy is not None:
            kwargs["caching_strategy"] = caching_strategy
        return ConfigVariableNode(node_class=cls, config_path=config_path, **kwargs)


class AliasedVariableNode(VariableNode):
    def __init__(self, parent_node: VariableNode):
        super().__init__()
        self.parent_node = parent_node

    def _generate_nodes(
        self, name: str, config: "typing.Dict[str, typing.Any]"
    ) -> "typing.List[node.Node]":
        # TODO this should output a list of functions that take in the outputs of one variable node and return them with a different name (do not redo compute)
        raise NotImplementedError()

    def alias(self) -> "AliasedVariableNode":
        # Reroute alias viability check to parent node
        return self.parent_node.alias()


class CloneVariableNode(VariableNode):
    def __init__(self, parent_node: VariableNode):
        super().__init__()
        self.parent_node = parent_node

    def _generate_nodes(
        self, name: str, config: "typing.Dict[str, typing.Any]"
    ) -> "typing.List[node.Node]":
        return self.parent_node._generate_nodes(name, config)

    def alias(self) -> "AliasedVariableNode":
        # Reroute alias viability check to parent node
        return self.parent_node.alias()
