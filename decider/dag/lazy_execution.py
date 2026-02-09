from dataclasses import dataclass, field
import polars as pl
import typing as t
from hamilton.lifecycle.base import BasePostGraphConstruct
from collections import OrderedDict

if t.TYPE_CHECKING:
    from hamilton import graph, node
    from polars._typing import SchemaDict

@dataclass
class Step:
    func: callable
    inputs: list[str]


@dataclass
class EmptyExecutionGraph:
    steps = None
    static_data: dict[str, t.Any] = field(default_factory=dict)

    def merge(self, other: "LazyExecutionGraph") -> "LazyExecutionGraph":
        merged_static = {**self.static_data, **other.static_data}
        return LazyExecutionGraph(
            steps=other.steps,
            static_data=merged_static,
            input_data=other.input_data,
            input_name=other.input_name
        )


@dataclass
class LazyExecutionGraph:
    steps: OrderedDict[str, Step]
    static_data: dict[str, t.Any]
    input_data: pl.LazyFrame
    input_name: str

    def compute(self, **kwargs) -> pl.LazyFrame:
        """Execute the lazy graph with the provided input DataFrame."""
        # Note dependencies are resolved by the order of steps as hamilton ensures correct ordering
        data = {**self.static_data, **kwargs}
        
        final_step = None
        for step_name, step in self.steps.items():
            input_values = {input_name: data[input_name] for input_name in step.inputs}
            data[step_name] = step.func(**input_values)
            final_step = step_name
            
        return data[final_step]
    
    def merge(self, other: "LazyExecutionGraph") -> "LazyExecutionGraph":
        """Merge another LazyExecutionGraph into this one."""
        # e.g. [a,d,c,q] + [a,b,d,c]
        # we have two graphs here and we know they can both execute in order
        # [a,d,c,q] can execute in order and [a,b,d,c] can execute in order, 
        # we want to merge them into a single graph that can execute in order
        # without any duplicates.
        # so we can go through the steps of the first graph and add them to the merged graph,
        # then we go through the steps of the second graph and add them if they are not already present.
        # [a,d,c,q] + [a,b,d,c] -> [a,d,c,q,b]
        merged_steps = OrderedDict(self.steps)
        merged_steps.update((k, v) for k, v in other.steps.items() if k not in self.steps)

        merged_static = {**self.static_data, **other.static_data}
        
        return LazyExecutionGraph(
            steps=merged_steps,
            static_data=merged_static,
            input_data=self.input_data,
            input_name=self.input_name
        )



    
def wrap_input_node(node_obj: "node.Node") -> "node.Node":
    """Wrap input node to return LazyExecutionGraph with input step."""
    original_callable = node_obj.callable
    assert len(node_obj.input_types) == 1, "Input node must have exactly one input"
    external_input_name = list(node_obj.input_types.keys())[0]

    
    def wrapped_input(**kwargs: pl.DataFrame) -> LazyExecutionGraph:
        """Input node that creates initial LazyExecutionGraph."""
        return LazyExecutionGraph(
            steps=OrderedDict([
                (node_obj.name, Step(
                    func=original_callable,
                    inputs=[external_input_name]
                ))
            ]),
            static_data={},
            input_data=kwargs[external_input_name],
            input_name=external_input_name,
        )
    
    return node_obj.copy_with(
        callabl=wrapped_input,
    )

def wrap_processing_node(node_obj: "node.Node") -> "node.Node":
    """Wrap processing nodes to add steps to LazyExecutionGraph."""
    original_callable = node_obj.callable
    
    def wrapped_processing(**kwargs) -> LazyExecutionGraph:
        """Processing node that adds step to LazyExecutionGraph."""
        lazy_graph = EmptyExecutionGraph()
        
        # Separate lazy graphs from static values
        for param_name, value in kwargs.items():
            if isinstance(value, LazyExecutionGraph):
                lazy_graph = lazy_graph.merge(value)
            else:
                lazy_graph.static_data[param_name] = value
        
        # If there is no lazy compute just compute eagerly and return as static value
        if lazy_graph.steps is None:
            return original_callable(**kwargs)

        lazy_graph.steps[node_obj.name] = Step(
            func=original_callable,
            inputs=list(node_obj.input_types.keys())
        )
        return lazy_graph
    
    wrapped_processing.__name__ = node_obj.name
    wrapped_processing.__doc__ = f"Lazy processing node for {node_obj.name}"
    
    return node_obj.copy_with(
        callabl=wrapped_processing,
    )

def wrap_collector_node(
    node_obj: "node.Node", 
    schema: "SchemaDict", 
    **kwargs
) -> "node.Node":
    """Wrap collector node (renamed output) to add final collection step."""
    node_obj = wrap_processing_node(node_obj)
    original_callable = node_obj.callable
    _map_batches_kwargs = kwargs

    def wrapped_collector(**kwargs) -> LazyExecutionGraph:
        """Collector node that adds final collection step."""
        nonlocal _map_batches_kwargs
        result = original_callable(**kwargs)
        if not isinstance(result, LazyExecutionGraph):
            return result
        return result.input_data.map_batches(
            lambda df: pl.from_pandas(result.compute(**{result.input_name: df})),
            schema=schema,
            **_map_batches_kwargs
        )

    
    wrapped_collector.__name__ = node_obj.name
    wrapped_collector.__doc__ = f"Lazy collector node for {node_obj.name}"
    
    return node_obj.copy_with(
        callabl=wrapped_collector,
    )
