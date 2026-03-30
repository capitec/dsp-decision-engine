import typing as t
import inspect
from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from decider.types import TInputType, TOutputType
from decider._ext import TypeDiscriminatedBaseModule

if t.TYPE_CHECKING:
    from decider.modules.primitives.mapper import MapperModule, ModuleInputSelector, ModuleOutputSelector

@dataclass
class StaticValueNode:
    value: t.Any

    @property
    def node_id(self) -> None:
        return None

    def __call__(self, inputs, cache=None) -> t.Any:
        return self.value
    
@dataclass
class ExternalInputNode:
    input_name: str

    @property
    def node_id(self) -> None:
        return None

    def __call__(self, inputs, cache= None) -> t.Any:
        return inputs[self.input_name]
    

@dataclass
class Node:
    name: str
    callable: t.Callable
    namespace: t.Tuple[str, ...] = field(default_factory=tuple)
    
    input_map: t.Dict[str, t.Union["Node", StaticValueNode, ExternalInputNode]] = field(
        default_factory=dict, 
        metadata={
            "description": "Maps this module's input parameters to external variable names. "
                         "Format: {my_input_param: external_variable_name}. "
                         "Example: {'data': 'user_records'} means this module's 'data' parameter "
                         "will receive the value from the external 'user_records' variable."
        }
    )

    @property
    def node_id(self) -> t.Tuple[str, ...]:
        return self.namespace + (self.name,)

    @classmethod
    def from_callable(
        cls, 
        func: t.Callable,
        name: t.Optional[str] = None,
        input_map: t.Optional[t.Dict[str, str]] = None,
        static_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> "Node":
        """Create a DeciderNode from a callable function.
        
        Args:
            func: The callable function to wrap in a DeciderNode
            name: Optional node name. If not provided, uses func.__name__
            input_map: Optional mapping of function parameters to external variable names.
                      Format: {function_param: external_variable_name}.
                      If not provided, all required parameters map to themselves.
            static_kwargs: Optional static keyword arguments to inject into the function
            
        Returns:
            A DeciderNode instance configured with the function and mappings
            
        Example:
            def process_data(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
                return data[data.value > threshold]
                
            # Create node that maps 'data' param to 'user_records' variable
            node = DeciderNode.from_callable(
                process_data,
                name="data_processor", 
                input_map={"data": "user_records"},
                static_kwargs={"threshold": 100}  # Inject constant threshold
            )
        """
        name = name or func.__name__
        function_kwargs = inspect.signature(func).parameters
        static_kwargs = static_kwargs or {}
        has_var_keyword = any(
            v.kind == inspect.Parameter.VAR_KEYWORD
            for v in function_kwargs.values()
        )
        named_params = {
            k
            for k, v in function_kwargs.items()
            if v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        required_kwargs = {
            k
            for k in named_params
            if function_kwargs[k].default == inspect.Parameter.empty
            and k not in static_kwargs
        }
        input_map = input_map or {}
        # Base map: all required named params (use provided mapping or identity)
        resolved_map = {k: input_map.get(k, k) for k in required_kwargs}
        # Extra keys in input_map that don't correspond to any named param
        extra_map_keys = {k for k in input_map if k not in named_params and k not in static_kwargs}
        if extra_map_keys:
            if not has_var_keyword:
                raise ValueError(
                    f"Parameters {sorted(extra_map_keys)} are in input_map but have no matching "
                    f"parameter in '{func.__name__}' and the function has no **kwargs to absorb them."
                )
            # Forward them through **kwargs
            for k in extra_map_keys:
                resolved_map[k] = input_map[k]
        input_map = resolved_map
        return cls(
            name=name,
            callable=func,
            input_map=(
                {k: ExternalInputNode(input_name=input_map[k]) for k in input_map}|
                {k: StaticValueNode(value=static_kwargs[k]) for k in static_kwargs}
            ),
        )


class BaseModule(TypeDiscriminatedBaseModule, ABC):
    name: str

    def as_graph(self):
        ...

    @property
    def outputs(self):
        from decider.modules.primitives.mapper import ModuleOutputAccessor
        return ModuleOutputAccessor(self)

    @property
    def inputs(self):
        from decider.modules.primitives.mapper import ModuleInputAccessor
        return ModuleInputAccessor(self)

    @property
    def output_names(self) -> t.List[str]:
        nodes = self.expand_nodes()
        referenced_nodes = set()
        
        for node in nodes:
            for input_ref in node.input_map.values():
                if isinstance(input_ref, Node):
                    referenced_nodes.add(input_ref.node_id)
        
        output_nodes = [node for node in nodes if node.node_id not in referenced_nodes]
        return [node.name for node in output_nodes]

    @property
    def input_names(self) -> t.List[str]:
        nodes = self.expand_nodes()
        external_inputs = set()
        
        for node in nodes:
            for input_ref in node.input_map.values():
                if isinstance(input_ref, ExternalInputNode):
                    external_inputs.add(input_ref.input_name)
        
        return sorted(external_inputs)

    def __or__(self, other: t.Union["BaseModule", "ModuleInputSelector"]) -> "MapperModule":
        from decider.modules.primitives.mapper import MapperModule, ModuleInputSelector
        
        if isinstance(other, ModuleInputSelector):
            my_outputs = self.output_names
            if len(my_outputs) != 1:
                raise ValueError(
                    f"Module '{self.name}' has {len(my_outputs)} outputs ({my_outputs}). "
                    f"Use .outputs.<name> to select one explicitly."
                )
            
            if isinstance(self, MapperModule):
                new_modules = self.modules + [other.module]
                new_mappings = dict(self.mappings)
                if other.module.name not in new_mappings:
                    new_mappings[other.module.name] = {}
                new_mappings[other.module.name][other.input_name] = (self.name, my_outputs[0])
                return self.model_copy(update={"modules": new_modules, "mappings": new_mappings})
            else:
                return MapperModule(
                    name=self.name,
                    modules=[self, other.module],
                    mappings={other.module.name: {other.input_name: (self.name, my_outputs[0])}},
                )
        
        elif isinstance(other, BaseModule):
            from decider.modules.primitives.mapper import MapperModule as MapperClass
            
            my_outputs = self.output_names
            other_inputs = other.input_names
            
            if len(my_outputs) != 1:
                raise ValueError(
                    f"Module '{self.name}' has {len(my_outputs)} outputs ({my_outputs}). "
                    f"Use .outputs.<name> | other.inputs.<name> for explicit wiring."
                )
            
            if len(other_inputs) != 1:
                raise ValueError(
                    f"Module '{other.name}' has {len(other_inputs)} inputs ({other_inputs}). "
                    f"Use self.outputs.<name> | other.inputs.<name> for explicit wiring."
                )
            
            if isinstance(self, MapperClass):
                if isinstance(other, MapperClass):
                    new_modules = self.modules + other.modules
                    new_mappings = dict(self.mappings)
                    new_mappings.update(other.mappings)
                else:
                    new_modules = self.modules + [other]
                    new_mappings = dict(self.mappings)
                
                if other.name not in new_mappings:
                    new_mappings[other.name] = {}
                new_mappings[other.name][other_inputs[0]] = (self.name, my_outputs[0])
                return self.model_copy(update={"modules": new_modules, "mappings": new_mappings})
            else:
                return MapperClass(
                    name=self.name,
                    modules=[self, other],
                    mappings={other.name: {other_inputs[0]: (self.name, my_outputs[0])}},
                )
        
        else:
            raise TypeError(f"Cannot wire BaseModule to {type(other).__name__}")

    def __lshift__(self, mapping: t.Dict[str, t.Union["BaseModule", "ModuleOutputSelector"]]) -> "MapperModule":
        from decider.modules.primitives.mapper import MapperModule as MapperClass, ModuleOutputSelector
        
        resolved_mappings = {}
        modules_to_add = []
        extra_mappings = {}
        
        for input_var_name, source in mapping.items():
            if isinstance(source, ModuleOutputSelector):
                resolved_mappings[input_var_name] = (source.module.name, source.output_node_name)
                if source.module not in modules_to_add:
                    modules_to_add.append(source.module)
            elif isinstance(source, BaseModule):
                if isinstance(source, MapperClass):
                    modules_to_add.extend(source.modules)
                    extra_mappings.update(source.mappings)
                    source_outputs = source.output_names
                    if len(source_outputs) != 1:
                        raise ValueError(
                            f"Module '{source.name}' has {len(source_outputs)} outputs ({source_outputs}). "
                            f"Use module.outputs.<name> to select one explicitly."
                        )
                    resolved_mappings[input_var_name] = (source.name, source_outputs[0])
                else:
                    source_outputs = source.output_names
                    if len(source_outputs) != 1:
                        raise ValueError(
                            f"Module '{source.name}' has {len(source_outputs)} outputs ({source_outputs}). "
                            f"Use module.outputs.<name> to select one explicitly."
                        )
                    resolved_mappings[input_var_name] = (source.name, source_outputs[0])
                    if source not in modules_to_add:
                        modules_to_add.append(source)
            else:
                raise TypeError(f"Mapping value must be BaseModule or ModuleOutputSelector, got {type(source).__name__}")
        
        if isinstance(self, MapperClass):
            new_modules = modules_to_add + self.modules
            new_mappings = dict(self.mappings)
            new_mappings.update(extra_mappings)
            if self.name not in new_mappings:
                new_mappings[self.name] = {}
            new_mappings[self.name].update(resolved_mappings)
            return self.model_copy(update={"modules": new_modules, "mappings": new_mappings})
        else:
            return MapperClass(
                name=self.name,
                modules=modules_to_add + [self],
                mappings={self.name: resolved_mappings, **extra_mappings},
            )

    def bind(self, **kwargs: t.Union["BaseModule", "ModuleOutputSelector"]) -> "MapperModule":
        return self.__lshift__(kwargs)

    def execute(
        self, 
        inputs: TInputType,
        outputs: t.List[str] = None,
        **kwargs
    ) -> TOutputType:
        # TODO i really think that we should pass in an executor here and then do a build step and an execute step to keep thinks a bit seperated and modular
        # But for the sake of simplicity lets just keep this as is.
        pass
    
    @abstractmethod
    def expand_nodes(self) -> t.List[Node]:
        """Expands the module into a list of DeciderNodes."""
        ...

    def module_namespaced_nodes(
        self,
        module_name: t.Optional[str] = None,
    ) -> t.List[Node]:
        """
        Expand module nodes with namespace and apply input mapping.
        
        This method:
        1. Expands the module into DeciderNodes
        2. Adds the module namespace to each node
        3. Applies input mapping to redirect external variables to internal parameters
        
        Args:
            module_name: The namespace to apply to all nodes in this module
            
        Returns:
            List of DeciderNodes with namespace and input mapping applied
        """
        # Get the raw nodes from the module
        module_name = module_name or self.name
        raw_nodes = self.expand_nodes()

        return [
            replace(n, namespace=(module_name,) + n.namespace) for n in raw_nodes
        ]
