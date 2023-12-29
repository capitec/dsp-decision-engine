from types import ModuleType
from typing import Any, Dict, Union, Callable, List
from hamilton import driver, base, function_modifiers, node
from engine.components.calculate import VariableNodeExpander
from engine.util import get_variable_nodes, get_mod_outputs, get_direct_modules


def expand_variable_nodes(functions: List[VariableNodeExpander], config:dict, return_unprocessed: bool = False):
    res = []
    not_processed = []
    for f in functions:
        if not isinstance(f, VariableNodeExpander):
            not_processed.append(f)
            continue
        res.extend(f.generate_nodes(config))
    if return_unprocessed:
        return res, not_processed
    return res


class Driver(driver.Driver):
    def __init__(
        self,
        config: Dict[str, Any],
        *modules: ModuleType,
        adapter: base.HamiltonGraphAdapter = None,
        _graph_executor: driver.GraphExecutor = None,
        skip_create_final_vars: bool = False
    ):
        direct_modules = []
        for mod in modules:
            direct_modules.extend(get_direct_modules(mod))
        super().__init__(config, *modules, *direct_modules, adapter=adapter, _graph_executor=_graph_executor)

        # Enable the use of nodes like x = engine.calculate(lambda: y)
        variable_nodes = expand_variable_nodes(
            get_variable_nodes(modules+tuple(direct_modules)), 
            config
        )
        # remove all external nodes from graph (deps)
        self.graph.nodes = {
            k: v for 
            k, v in self.graph.nodes.items() 
            if v._node_source != node.NodeType.EXTERNAL
        }
        self.graph = self.graph.with_nodes({n.name: n for n in variable_nodes})
        outputs = []
        for mod in modules:
            outputs.extend(get_mod_outputs(mod))
        self.outputs = outputs

        if skip_create_final_vars:
            self.final_vars = None
        else:
            self.final_vars = self._create_final_vars(self.outputs)
    
        # Allow direct access to module inner functions
    def execute(
        self,
        final_vars: List[Union[str, Callable, driver.Variable]] = None,
        overrides: Dict[str, Any] = None,
        display_graph: bool = False,
        inputs: Dict[str, Any] = None,
    ) -> Any:
        final_vars = final_vars or []
        return super().execute(
            final_vars + self.outputs,
            overrides,
            display_graph,
            inputs
        )
    
    def raw_execute(
        self,
        final_vars: List[str] = None,
        overrides: Dict[str, Any] = None,
        display_graph: bool = False,
        inputs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if final_vars is None:
            final_vars = self._create_final_vars(self.outputs) if self.final_vars is None else self.final_vars
        return super().raw_execute(
            final_vars, # Dont do a concatenation as execute might already concatenate
            overrides,
            display_graph,
            inputs
        )

class subdag(function_modifiers.subdag):
    def __init__(
        self,
        *load_from: Union[ModuleType, Callable],
        inputs: Dict[str, function_modifiers.dependencies.ParametrizedDependency] = None,
        config: Dict[str, Any] = None,
        namespace: str = None,
        final_node_name: str = None,
        external_inputs: List[str] = None,
    ):
        """Adds a subDAG to the main DAG.

        :param load_from: The functions that will be used to generate this subDAG.
        :param inputs: Parameterized dependencies to inject into all sources of this subDAG.
            This should *not* be an intermediate node in the subDAG.
        :param config: A configuration dictionary for *just* this subDAG. Note that this passed in
            value takes precedence over the DAG's config.
        :param namespace: Namespace with which to prefix nodes. This is optional -- if not included,
            this will default to the function name.
        :param final_node_name: Name of the final node in the subDAG. This is optional -- if not included,
            this will default to the function name.
        :param external_inputs: Parameters in the function that are not produced by the functions
            passed to the subdag. This is useful if you want to perform some logic with other inputs
            in the subdag's processing function. Note that this is currently required to
            differentiate and clarify the inputs to the subdag.

        """
        super().__init__(
            *load_from,
            inputs=inputs,
            config=config,
            namespace=namespace,
            final_node_name=final_node_name,
            external_inputs=external_inputs
        )
        self.subdag_functions+= get_variable_nodes(load_from)

    @staticmethod
    def collect_nodes(config: Dict[str, Any], subdag_functions: List[Union[Callable, VariableNodeExpander]]) -> List[node.Node]:
        inner_subdag_functions = []
        nodes_tmp, inner_subdag_functions = expand_variable_nodes(subdag_functions, config, return_unprocessed=True)

        nodes = []
        for node_ in nodes_tmp:
            nodes.append(node_.copy_with(tags={**node_.tags, **function_modifiers.recursive.NON_FINAL_TAGS}))
        nodes += function_modifiers.subdag.collect_nodes(config, inner_subdag_functions)
        return nodes
