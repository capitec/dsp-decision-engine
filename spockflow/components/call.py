import typing
from types import ModuleType
from dataclasses import dataclass, field
from typing import Any
from hamilton import function_modifiers as fm
from hamilton import node, base, common
from spockflow.nodes import VariableNodeExpander
from spockflow.core import subdag
from spockflow._util import get_mod_outputs, get_output_types, init_module_variable_node_names

@dataclass
class ExtractFromNamespaceNode(VariableNodeExpander):
    namespace_node: typing.Union[VariableNodeExpander, str]
    output_key: str
    output_type: typing.Union[typing.Type, typing.Callable]

    @property
    def name(self):
        if self._name is None:
            self.set_name(self.output_key)
        return super().name
    def generate_nodes(self, config: dict, var_name: str=None) -> typing.List[node.Node]:
        self.set_name(var_name)
        def identity(**kwargs):
            return next(iter(kwargs.values()))
        
        namespace = self.namespace_node if isinstance(self.namespace_node, str) else self.namespace_node.name
        ns_key = fm.recursive.assign_namespace(self.output_key, namespace)
        output_type = self.output_type
        if isinstance(output_type , typing.Callable):
            output_type = output_type(ns_key)
        
        module_tags = {} if isinstance(self.namespace_node, str) else {"module": self.namespace_node.module}
        return [node.Node(
            name=self.name,
            typ=output_type,
            doc_string = f"Extract {namespace}.{self.output_key} as {self.name}",
            callabl=identity,
            node_source = node.NodeType.STANDARD,
            input_types = {ns_key: (output_type, node.DependencyType.REQUIRED)},
            tags = module_tags,
            namespace = (),
        )]

@dataclass
class SubdagDynamicNode(VariableNodeExpander):
    mods : typing.List[ModuleType]
    args: typing.List[typing.Any]
    kwargs: typing.Dict[str, typing.Any]
    _output_types: typing.Dict[str, typing.Type] = field(default_factory=dict)

    def get_inputs(self):
        res_inputs = {}
        for i,v in enumerate(self.args):
            if isinstance(v, typing.Callable):
                if v.__name__ in res_inputs: raise ValueError(f"Value: {v.__name__} specified more than once.")
                res_inputs[v.__name__] = fm.source(v.__name__)
                continue
            if isinstance(v, VariableNodeExpander):
                try:
                    dst_name = v.name
                except ValueError:
                    dst_name = None
                if dst_name is not None:
                    if dst_name in res_inputs: raise ValueError(f"Value: {dst_name} specified more than once.")
                    res_inputs[dst_name] = fm.source(v.name)
                    continue
            raise ValueError(f"Could not infer name of arg {i} with value {v}. Please specify name with kwarg.")
            
        for k,v in self.kwargs.items():
            if k in res_inputs: raise ValueError(f"Value: {v.__name__} specified more than once.")
            if isinstance(v, typing.Callable):
                res_inputs[k] = fm.source(v.__name__)
            elif isinstance(v, VariableNodeExpander):
                res_inputs[k] = fm.source(v.name)
            elif isinstance(v, fm.dependencies.SingleDependency):
                res_inputs[k] = v
            else:
                res_inputs[k] = fm.value(v)
        return res_inputs


    def generate_nodes(self, config: dict, var_name: str=None) -> typing.List[node.Node]:
        self.set_name(var_name)
        # Get all outputs in all modules ordered by order of modules inputted
        outputs = []
        for mod in self.mods:
            try:
                mod_outs = get_mod_outputs(mod)
                # TODO not sure if its needed to do a duplicate check here.
                # Think it will be taken care of by the hamiltonian graph
                outputs.extend(mod_outs)
            except AssertionError:
                pass

        # Ensure all variable nodes have initialized names
        init_module_variable_node_names(self.mods)
        # Get outputs as strings
        module_set = {_module.__name__ for _module in self.mods}
        output_tokens = common.convert_output_values(outputs, module_set)
        output_types = get_output_types(outputs)

        # get output types
        subdag_inputs = self.get_inputs()
        # Ensure config items are overwritten by subdag_inputs
        subdag_config = {k:c for k,c in config.items() if k not in subdag_inputs}
        @subdag(*self.mods, inputs=subdag_inputs, config=subdag_config, namespace=self.name)
        def _extract_outputs_() -> dict:
            # Mock function to allow subdag to expand as expected
            pass

        nodes: typing.List[node.Node] = fm.base.resolve_nodes(_extract_outputs_, config)
        output_node = [(i,n) for i, n in enumerate(nodes) if n.name == "_extract_outputs_"]
        assert len(output_node) == 1, "This should always be the case"
        output_node_idx, output_node = output_node[0]
        new_input_types = {
            fm.recursive.assign_namespace(k, self.name): (v, node.DependencyType.REQUIRED) 
            for k,v in zip(output_tokens, output_types)
        }
        
        def _extract_outputs_(**kwargs: dict) -> dict:
            return kwargs
        
        nodes[output_node_idx] = output_node.copy_with(
            callabl=_extract_outputs_,
            input_types=new_input_types,
            name=self.name,
            tags={"module": self.module}
        )

        # TODO see if this causes more issues than it is worth
        self._output_types.update({n.name: n.type for n in nodes})
        return nodes
    
    def get_output_type(self, key: str):
        if key not in self._output_types: return typing.Any
        return self._output_types[key]
    
    def __getattr__(self, __name: str) -> Any:
        return ExtractFromNamespaceNode(
            namespace_node=self,
            output_key=__name,
            output_type=self.get_output_type
        )


def call_mods(mods, *args, **kwargs):
    return SubdagDynamicNode(mods=mods, args=args, kwargs=kwargs)

T = typing.TypeVar('T')
def call(mod: T, *args, **kwargs) -> T:
    return call_mods([mod], *args, **kwargs)
