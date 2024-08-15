import inspect
from typing import Callable, List, Union, Collection, TYPE_CHECKING, Dict, Any

from hamilton import node, driver
from hamilton.function_modifiers import subdag
from hamilton.function_modifiers import base

from spockflow.nodes import VariableNode

if TYPE_CHECKING:
    from types import ModuleType
    from hamilton import graph


class Driver(driver.Driver):
    def raw_execute(
        self,
        final_vars: List[str] = None,
        overrides: Dict[str, Any] = None,
        display_graph: bool = False,
        inputs: Dict[str, Any] = None,
        _fn_graph: "graph.FunctionGraph" = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        function_graph = _fn_graph if _fn_graph is not None else self.graph
        if final_vars is None:
            final_vars = configure_output._get_outputs(function_graph)
        return super().raw_execute(
            final_vars=final_vars,
            overrides=overrides,
            display_graph=display_graph,
            inputs=inputs,
            _fn_graph=function_graph,
            *args,
            **kwargs,
        )

    def execute(
        self,
        final_vars: List[Union[str, Callable, "Variable"]] = None,
        overrides: Dict[str, Any] = None,
        display_graph: bool = False,
        inputs: Dict[str, Any] = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        if final_vars is None:
            final_vars = configure_output._get_outputs(self.graph)
        return super().execute(
            final_vars=final_vars,
            overrides=overrides,
            display_graph=display_graph,
            inputs=inputs,
            *args,
            **kwargs,
        )


class configure_output(subdag):
    DEFAULT_OUTPUT_TAG = "IsSpockDefaultOutput"

    @staticmethod
    def is_variable_node(v):
        return isinstance(v, VariableNode)

    @classmethod
    def get_variable_nodes(cls, m: "ModuleType"):
        var_nodes_w_names: "List[VariableNode]" = inspect.getmembers(
            m, predicate=cls.is_variable_node
        )
        return [
            var_node._set_name(name)._set_module(m)
            for name, var_node in var_nodes_w_names
        ]

    def __init__(
        self,
        included_modules: "Union[ModuleType, Callable, VariableNode]",
        ignore_output=False,
        output_names: List[str] = None,
    ):
        self.output_names = set(output_names) if output_names is not None else set()
        self.ignore_output = ignore_output
        if included_modules:
            self.do_generate = True
            super().__init__(*included_modules)

            for m in included_modules:
                # Only add modules
                if isinstance(m, Callable):
                    continue
                if isinstance(m, VariableNode):
                    self.subdag_functions.append(m)
                self.subdag_functions += self.get_variable_nodes(m)
        else:
            self.config = {}
            self.inputs = {}
            self.subdag_functions = []
            self.do_generate = True

    def set_spock_output_flag(self, n: node.Node, force: bool = False) -> node.Node:
        if force or n.name in self.output_names:
            n.add_tag(self.DEFAULT_OUTPUT_TAG, True)
        return n

    @classmethod
    def _get_outputs(cls, g: "graph.FunctionGraph") -> List[str]:
        if not hasattr(g, "_spock_cached_default_out"):
            default_out = []
            for k, n in g.nodes.items():
                if n.tags.get(cls.DEFAULT_OUTPUT_TAG, False):
                    default_out.append(k)
            g._spock_cached_default_out = default_out
        return g._spock_cached_default_out

    def generate_nodes(
        self, fn: "Callable", configuration: "Dict[str, Any]"
    ) -> "Collection[node.Node]":
        this_module = inspect.getmodule(fn)
        subdag_functions = self.subdag_functions + self.get_variable_nodes(this_module)
        # Resolve all nodes from passed in functions
        resolved_config = dict(configuration, **self.config)
        nodes = []
        for sd_fn in subdag_functions:
            for node_ in base.resolve_nodes(sd_fn, resolved_config):
                nodes.append(self.set_spock_output_flag(node_))
        # TODO might hook in here to add config
        # nodes += self._create_additional_static_nodes(nodes, namespace)
        # Add the final node that does the translation
        # nodes += [self.add_final_node(fn, final_node_name, namespace)]
        if not self.ignore_output:
            nodes.append(self.set_spock_output_flag(node.Node.from_fn(fn), force=True))
        return nodes


def initialize_spock_module(
    module_name, included_modules=None, output_names: List[str] = None
):
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

    @configure_output(included_modules, ignore_output=True, output_names=output_names)
    def spock_bootstrap_entrypoint_fn__() -> None:
        pass

    spock_bootstrap_entrypoint_fn__.__module__ = module_name
    setattr(
        caller_module,
        spock_bootstrap_entrypoint_fn__.__name__,
        spock_bootstrap_entrypoint_fn__,
    )
