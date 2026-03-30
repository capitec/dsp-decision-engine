import typing as t
import inspect
from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from decider.types import TInputType, TOutputType
from decider._ext import TypeDiscriminatedBaseModule

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
