import typing as t
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decider.modules.core import Node

from decider.modules.core import BaseModule

class ModuleOutputSelector:
    def __init__(self, module: BaseModule, output_node_name: str):
        self.module = module
        self.output_node_name = output_node_name

    def __or__(self, other: "ModuleInputSelector") -> "MapperModule":
        if not isinstance(other, ModuleInputSelector):
            raise TypeError(
                f"Can only wire ModuleOutputSelector to ModuleInputSelector, "
                f"got {type(other).__name__}"
            )
        return MapperModule(
            name=self.module.name,
            modules=[self.module, other.module],
            mappings={
                other.module.name: {
                    other.input_name: (self.module.name, self.output_node_name)
                }
            },
        )


class ModuleInputSelector:
    def __init__(self, module: BaseModule, input_name: str):
        self.module = module
        self.input_name = input_name


class ModuleOutputAccessor:
    def __init__(self, module: "BaseModule"):
        self._module = module

    def __getattr__(self, name: str) -> ModuleOutputSelector:
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")
        return ModuleOutputSelector(module=self._module, output_node_name=name)

    def __getitem__(self, name: str) -> ModuleOutputSelector:
        return ModuleOutputSelector(module=self._module, output_node_name=name)


class ModuleInputAccessor:
    def __init__(self, module: "BaseModule"):
        self._module = module

    def __getattr__(self, name: str) -> ModuleInputSelector:
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")
        return ModuleInputSelector(module=self._module, input_name=name)

    def __getitem__(self, name: str) -> ModuleInputSelector:
        return ModuleInputSelector(module=self._module, input_name=name)


class MapperModule(BaseModule):
    type: t.Literal["mapper"]
    name: str = ""
    modules: t.List[BaseModule] = field(default_factory=list)
    mappings: t.Dict[str, t.Dict[str, t.Tuple[str, str]]] = field(default_factory=dict)

    def expand_nodes(self) -> t.List["Node"]:
        from decider.modules.core import ExternalInputNode
        
        module_registry = {}
        for module in self.modules:
            if module.name in module_registry:
                raise ValueError(f"Duplicate module name: '{module.name}'")
            module_registry[module.name] = module

        node_registry = {}
        all_nodes = []
        
        for module in self.modules:
            namespaced_nodes = module.module_namespaced_nodes()
            for node in namespaced_nodes:
                node_registry[node.node_id] = node
                all_nodes.append(node)

        for dest_module_name, input_mappings in self.mappings.items():
            for input_var_name, (src_module_name, src_output_node_name) in input_mappings.items():
                src_node_id = (src_module_name, src_output_node_name)
                
                if src_node_id not in node_registry:
                    raise ValueError(
                        f"Source node '{src_module_name}.{src_output_node_name}' not found "
                        f"in node registry. Available nodes: {list(node_registry.keys())}"
                    )
                
                source_node = node_registry[src_node_id]
                
                for node in all_nodes:
                    if not node.node_id[0] == dest_module_name:
                        continue
                    
                    new_input_map = {}
                    modified = False
                    
                    for param_name, input_ref in node.input_map.items():
                        if isinstance(input_ref, ExternalInputNode) and input_ref.input_name == input_var_name:
                            new_input_map[param_name] = source_node
                            modified = True
                        else:
                            new_input_map[param_name] = input_ref
                    
                    if modified:
                        node.input_map = new_input_map

        return all_nodes

    def module_namespaced_nodes(self, module_name: t.Optional[str] = None) -> t.List["Node"]:  # noqa: ARG002
        return self.expand_nodes()

    @property
    def output_names(self) -> t.List[str]:
        nodes = self.expand_nodes()
        referenced_nodes = set()
        
        for node in nodes:
            from decider.modules.core import Node as NodeClass
            for input_ref in node.input_map.values():
                if isinstance(input_ref, NodeClass):
                    referenced_nodes.add(input_ref.node_id)
        
        output_nodes = [node for node in nodes if node.node_id not in referenced_nodes]
        return [node.name for node in output_nodes]

    @property
    def input_names(self) -> t.List[str]:
        from decider.modules.core import ExternalInputNode
        
        nodes = self.expand_nodes()
        external_inputs = set()
        
        for node in nodes:
            for input_ref in node.input_map.values():
                if isinstance(input_ref, ExternalInputNode):
                    external_inputs.add(input_ref.input_name)
        
        return sorted(external_inputs)

    @property
    def outputs(self) -> ModuleOutputAccessor:
        return ModuleOutputAccessor(self)

    @property
    def inputs(self) -> ModuleInputAccessor:
        return ModuleInputAccessor(self)
