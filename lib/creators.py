import typing
import importlib
from types import ModuleType
from hamilton.function_modifiers import subdag
from hamilton.function_modifiers import base
from hamilton.function_modifiers.recursive import assign_namespace
from hamilton import (node, graph_utils)
from hamilton.function_modifiers.dependencies import (ParametrizedDependency,)

from lib import builtin_funcs

class parametrized_input_output(base.NodeCreator):
    @staticmethod
    def resolve_input_from_config(value_name, var):
        detected_type = typing.Any
        detected_dep_type = node.DependencyType.REQUIRED
        if isinstance(var, str):
            pass
        elif isinstance(var, dict):
            # value_name = var["value"]
            #TODO maybe use pydoc.locate
            detected_type = eval(var.get("type","None")) or detected_type
            if not var.get("required",True): detected_dep_type = node.DependencyType.OPTIONAL
        else:
            raise ValueError()
        return value_name, (detected_type, detected_dep_type)


    def generate_nodes(self, fn: typing.Callable, config: typing.Dict[str, typing.Any]) -> typing.List[node.Node]:
        created_node = node.Node.from_fn(fn)
        if "kwargs" not in created_node.input_types:
            raise ValueError(f"Node definition must contain **kwargs")
        new_inputs = {k:v for k,v in created_node.input_types.items() if k != "kwargs"}
        # TODO change this to model call and automatically infer typing.any and required true
        additional_inputs = dict(self.resolve_input_from_config(value_name, value_config) for value_name, value_config in config["inputs"].items())
        if not new_inputs.keys().isdisjoint(additional_inputs.keys()):
            raise ValueError(f"Can only override inputs that are not in the function definition.")
        new_inputs.update(additional_inputs)

        output_type = eval(config.get("output_type","None")) or created_node.type
        return [created_node.copy_with(typ=output_type, input_types=new_inputs)]
    
    def required_config(self) -> typing.Optional[typing.List[str]]:
        return ["inputs"]

    def optional_config(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        return {}

    def validate(self, fn: typing.Callable):
        pass


class subdag_multi_out(subdag):
    def __init__(
        self,
        *load_from: typing.Union[ModuleType, typing.Callable],
        inputs: typing.Dict[str, ParametrizedDependency] = None,
        config: typing.Dict[str, typing.Any] = None,
        namespace: str = None,
        output_node_map: typing.Dict[str, str] = None,
        external_inputs: typing.List[str] = None,
        isolate_inputs: bool = True
    ):
        output_node_map = output_node_map or {}
        root_node_name = next((key for key, value in output_node_map.items() if value == "__root__"), None)
        
        super().__init__(
            *load_from,
            inputs=inputs,
            config=config,
            namespace=namespace,
            final_node_name=root_node_name,
            external_inputs=external_inputs
        )
        self.output_node_map = output_node_map
        self.isolate_inputs = isolate_inputs

    @classmethod
    def get_rename_node(cls, internal_node: node.Node, external_name: str):
        return node.Node(
            external_name,
            internal_node.type,
            f"Output {internal_node.name} as {external_name}",
            callabl=lambda **kwargs: next(iter(kwargs.values())),
            tags={**internal_node.tags},
            node_source=node.NodeType.STANDARD,
            input_types={internal_node.name: (internal_node.type, node.DependencyType.REQUIRED)}
        )
    
    @classmethod
    def add_output_nodes(cls, namespace: str, output_node_map: typing.Dict[str, str], nodes: node.Node):
        namespaced_node_map = {
            assign_namespace(internal, namespace): external
            for external, internal in output_node_map.items()
            if internal != "__root__"
        }
        output_nodes = []
        exposed_nodes = []
        for node in nodes:
            if node.name in namespaced_node_map:
                output_nodes.append(cls.get_rename_node(node, namespaced_node_map[node.name]))
                exposed_nodes.append(node.name)

        unexposed_nodes = set(namespaced_node_map.keys())-set(exposed_nodes)
        if len(unexposed_nodes) > 0:
            raise ValueError(f"Could not expose nodes {unexposed_nodes}")
        return output_nodes
    
    @classmethod
    def validate_isolation(cls, nodes, namespace, allowed_inputs):
        namespace_prefix = assign_namespace(namespace, "")
        for node in nodes:
            missing_keys = set()
            for input_key in node.input_types.keys():
                if not input_key.startswith(namespace_prefix) or not input_key in allowed_inputs:
                    missing_keys.add(input_key)
            if len(missing_keys):
                raise ValueError(f"Missing inputs to subdag: {missing_keys}")
    
    def generate_nodes(
            self, 
            fn: typing.Optional[typing.Callable], 
            configuration: typing.Dict[str, typing.Any]
        ) -> typing.Collection[node.Node]:

        # Resolve all nodes from passed in functions
        resolved_config = dict(configuration, **self.config)
        nodes = self.collect_nodes(config=resolved_config, subdag_functions=self.subdag_functions)
        # Derive the namespace under which all these nodes will live
        if fn is not None:
            final_node_name = self._derive_name(fn)
            namespace = self._derive_namespace(fn)
        else:
            assert self.namespace is not None, "Namespace cannot be none when not defined over a function"
            namespace = self.namespace

        # Rename them all to have the right namespace
        nodes = self.add_namespace(nodes, namespace, self.inputs, self.config)
        # Create any static input nodes we need to translate
        nodes += self._create_additional_static_nodes(nodes, namespace)
        # Ensure all created nodes are isolated
        if self.isolate_inputs:
            self.validate_isolation(nodes, namespace, self.external_inputs)
        # Add the final node that does the translation
        if fn is not None:
            nodes += [self.add_final_node(fn, final_node_name, namespace)]
        nodes += self.add_output_nodes(namespace, output_node_map=self.output_node_map, nodes=nodes)
        return nodes


class config_dag(subdag_multi_out):

    def __init__(
        self,
        inputs: typing.Dict[str, ParametrizedDependency] = None,
        config: typing.Dict[str, typing.Any] = None,
    ):
        def dummy_fn():
            pass
        super().__init__(dummy_fn, config=config)
    
    @staticmethod
    def collect_functions(
        load_from: typing.Union[typing.Collection[ModuleType], typing.Collection[typing.Callable]]
    ) -> typing.List[typing.Callable]:
        return []

    @classmethod
    def collect_nodes(
        cls,
        config: typing.Dict[str, typing.Any], 
        subdag_functions: typing.List[typing.Callable]
    ) -> typing.List[node.Node]:
        from lib import primitives
        
        modules = [builtin_funcs]
        modules += [importlib.import_module(m) for m in config["imports"]]
        functions = sum([graph_utils.find_functions(module) for module in modules], [])
        function_map = dict(functions)

        nodes = []

        for primitive_id, primitive_expander in primitives.PRIMITIVES.items():
            for primitive_name, primitive_config in config.get(primitive_id, {}).items():
                nodes.extend(primitive_expander.expand_nodes(
                    primitive_name, 
                    primitive_config, 
                    function_map=function_map, 
                    config=config
                ))

        return nodes
    
    def _derive_namespace(self, configuration: typing.Dict[str, typing.Any]) -> str:
        """Utility function to derive a namespace from a function.

        :param fn: Function we're decorating.
        :return: The function we're outputting.
        """
        return configuration.get("namespace", "body")
        # return fn.__name__ if self.namespace is None else self.namespace
    
    def generate_nodes(self, fn: typing.Callable, configuration: typing.Dict[str, typing.Any]) -> typing.Collection[node.Node]:
        # Resolve all nodes from passed in functions
        resolved_config = dict(configuration, **self.config)
        nodes = self.collect_nodes(config=resolved_config, subdag_functions=self.subdag_functions)
        # Derive the namespace under which all these nodes will live
        namespace = self._derive_namespace(resolved_config)
        # Rename them all to have the right namespace
        # inputs = resolved_config.get("inputs", [])
        inputs = {}

        nodes = self.add_namespace(nodes, namespace, inputs, self.config)
        # Create any static input nodes we need to translate
        nodes += self._create_additional_static_nodes(nodes, namespace)
        # Namespace Isolation
        if not resolved_config.get("override_namespace_isolation", False):
            self.validate_isolation(nodes, namespace, self.external_inputs)
        # Add the final node that does the translation
        output_node_map = configuration.get("outputs", {})
        nodes += self.add_output_nodes(namespace, output_node_map=output_node_map, nodes=nodes)

        return nodes