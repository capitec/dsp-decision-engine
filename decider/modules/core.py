import typing as t
import inspect
import polars as pl
from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from decider.types import TInputType, TOutputType
from decider._ext import TypeDiscriminatedBaseModule

if t.TYPE_CHECKING:
    from decider.modules.primitives.mapper import MapperModule, ModuleOutputSelector
    from decider.executor import Executor, CompiledDag
    from decider.pipeline import Pipeline, ForkPipeline

@dataclass(slots=True)
class StaticValueNode:
    value: t.Any

    @property
    def node_id(self) -> None:
        return None

    def __call__(self, inputs, cache=None) -> t.Any:
        return self.value

    def get_expr(self) -> t.Any:
        return pl.lit(self.value)

    def get_frame_value(self, _frames: t.Dict[str, t.Any]) -> t.Any:
        return self.value

@dataclass(slots=True)
class ExternalInputNode:
    input_name: str

    @property
    def node_id(self) -> None:
        return None

    def __call__(self, inputs, cache= None) -> t.Any:
        return inputs[self.input_name]

    def get_expr(self) -> pl.Expr:
        return pl.col(self.input_name)

    def get_frame_value(self, frames: t.Dict[str, t.Any]) -> t.Any:
        if self.input_name not in frames:
            raise ValueError(
                f"Frame '{self.input_name}' not found. Available: {list(frames.keys())}"
            )
        return frames[self.input_name]


@dataclass(slots=True)
class Node:
    """Represents a computation unit in the graph.

    Nodes can be either:
    - Expression nodes: pl.Expr → pl.Expr (add columns to existing frames)
    - Frame nodes: pl.LazyFrame → pl.LazyFrame (create new frames via joins, etc.)
    """
    name: str
    callable: t.Callable
    node_type: t.Literal["expression", "frame"] = "expression"
    """Type of node: 'expression' adds columns, 'frame' creates new frames"""

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

    target_frame: str = "input"
    """For expression nodes: which frame to add columns to. For frame nodes: output frame name."""

    @property
    def node_id(self) -> t.Tuple[str, ...]:
        return self.namespace + (self.name,)

    def get_expr(self) -> pl.Expr:
        return pl.col(self.name)

    def get_input_expressions(self) -> t.Dict[str, t.Any]:
        return {param_name: ref.get_expr() for param_name, ref in self.input_map.items()}

    def get_frame_kwargs(self, frames: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        return {param_name: ref.get_frame_value(frames) for param_name, ref in self.input_map.items()}

    @property
    def frame_dependencies(self) -> t.List[str]:
        return [ref.input_name for ref in self.input_map.values() if isinstance(ref, ExternalInputNode)]

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
            node = Node.from_callable(
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

    def compile(
        self,
        executor: t.Optional["Executor"] = None,
        allow_overrides: bool = False,
    ) -> "CompiledDag":
        """Compile this module into an executable form.

        This is the compilation step - it converts nodes into ordered expressions.
        The result (CompiledDag) can be cached and reused for multiple executions.

        Args:
            executor: Optional executor override. If None, uses global default from settings.
            allow_overrides: If True, input columns whose names collide with this
                module's computed columns take precedence — the expression is skipped
                and the existing column is preserved. Use this for what-if injection
                (passing a pre-computed value to bypass the module's own logic).
                If False (default), a name collision raises a ValueError.

        Returns:
            CompiledDag ready for execution

        Example:
            >>> compiled = module.compile()           # raises on collision
            >>> compiled = module.compile(allow_overrides=True)  # injection mode
        """
        from decider.settings import get_default_executor

        exec_instance = executor or get_default_executor()
        nodes = self.expand_nodes()
        return exec_instance.compile(nodes, allow_overrides=allow_overrides)

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

    def __or__(self, other: t.Union["BaseModule", "Pipeline"]) -> "Pipeline":
        from decider.pipeline import Pipeline
        if isinstance(other, Pipeline):
            return Pipeline([self] + other.steps)
        return Pipeline([self, other])

    def __and__(self, other: t.Union["BaseModule", "Pipeline", "ForkPipeline"]) -> "ForkPipeline":
        from decider.pipeline import Pipeline, ForkPipeline
        left = Pipeline([self])
        right = other if isinstance(other, (Pipeline, ForkPipeline)) else Pipeline([other])
        return ForkPipeline(branches=[left, right])

    def include(self, *others: "BaseModule") -> "ForkPipeline":
        """Branch from self into parallel paths, one per module.

        engagement.include(mod1, mod2) is sugar for:
            (engagement | mod1) & (engagement | mod2)
        """
        from decider.pipeline import Pipeline, ForkPipeline
        branches = [Pipeline([self, other]) for other in others]
        return ForkPipeline(branches=branches)

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
        dataframes: t.Union[t.Dict[str, pl.LazyFrame], t.Dict[str, pl.DataFrame]],
        output_frames: t.Optional[t.List[str]] = None,
        executor: t.Optional["Executor"] = None,
        debug: bool = False,
        lazy: bool = True,
        allow_overrides: bool = False,
    ) -> t.Union[pl.LazyFrame, pl.DataFrame, t.Dict[str, pl.LazyFrame], t.Dict[str, pl.DataFrame]]:
        """Execute this module on input dataframes.

        Input frame convention:
            Pass a dict of frames keyed by name.  Use ``"input"`` as the key
            for the main frame — that is the default target every expression
            writes to::

                module.execute({"input": df})

            You may supply additional named frames for multi-frame modules
            (e.g. JoinModule) by adding extra keys::

                join_module.execute({"transactions": txn_df, "users": user_df})

        Args:
            dataframes: Input frames (name → LazyFrame or DataFrame).
            output_frames: If None, returns the final LazyFrame (or DataFrame
                when ``lazy=False``).  Pass a list of frame names to get a
                dict of results instead.
            executor: Optional executor override (uses default from settings if None).
            debug: If True, print debug information during execution.
            lazy: If True (default), results are returned as LazyFrame(s).
                Set to False to call ``.collect()`` automatically and return
                eager DataFrame(s).

        Returns:
            * ``lazy=True, output_frames=None``  → ``pl.LazyFrame``
            * ``lazy=False, output_frames=None`` → ``pl.DataFrame``
            * ``lazy=True, output_frames=[...]`` → ``Dict[str, pl.LazyFrame]``
            * ``lazy=False, output_frames=[...]`` → ``Dict[str, pl.DataFrame]``
        """
        from decider.pipeline import Pipeline
        result = Pipeline([self]).execute(
            dataframes,
            output_frames=output_frames,
            executor=executor,
            debug=debug,
            allow_overrides=allow_overrides,
        )
        if not lazy:
            if isinstance(result, dict):
                return {k: v.collect() for k, v in result.items()}
            return result.collect()
        return result

    @abstractmethod
    def expand_nodes(self) -> t.List[Node]:
        """Expands the module into a list of DeciderNodes.

        DEPRECATED: Use expand_expressions() instead.
        This method is kept for backward compatibility only.
        """
        ...

    def module_namespaced_nodes(self) -> t.List[Node]:
        """
        Expand module nodes with namespace and apply input mapping.

        This method:
        1. Expands the module into DeciderNodes
        2. Adds the module namespace to each node

        Returns:
            List of DeciderNodes with namespace and input mapping applied
        """
        raw_nodes = self.expand_nodes()
        return [
            replace(n, namespace=(self.name,) + n.namespace) for n in raw_nodes
        ]
