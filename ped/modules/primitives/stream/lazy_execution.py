"""Lazy execution wrappers for streaming PED nodes via polars map_batches.

These wrappers transform PED node callables so that execution is deferred and
batched using ``polars.LazyFrame.map_batches``, enabling efficient row-level
processing pipelines.

The execution model is:
  1. An *input node* receives a ``pl.LazyFrame`` and creates a
     ``LazyExecutionGraph`` that captures the computation steps.
  2. *Processing nodes* (nodes with ``pd.Series`` or ``pd.DataFrame`` inputs)
     accumulate steps into the graph instead of executing eagerly.
  3. A *collector node* finalises the graph by calling ``map_batches``, which
     applies all accumulated steps batch-by-batch on the LazyFrame.
"""
import typing as t
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd
import polars as pl

if t.TYPE_CHECKING:
    from polars._typing import SchemaDict


# ---------------------------------------------------------------------------
# Lazy execution graph primitives
# ---------------------------------------------------------------------------

class InputMapItem(t.NamedTuple):
    node_name: str  # The name of the output node providing the input
    param_name: str  # The name of the parameter to call the module with

@dataclass
class Step:
    func: t.Callable
    input_map: t.List[InputMapItem]  # mapping of step input name → function parameter name


@dataclass
class EmptyExecutionGraph:
    """Sentinel used when no lazy graph has been encountered yet."""

    static_data: t.Dict[str, t.Any] = field(default_factory=dict)

    @property
    def steps(self) -> t.Dict[str, Step]:
        return dict()

    def merge(self, other: "LazyExecutionGraph|EmptyExecutionGraph") -> "LazyExecutionGraph":
        merged_static = {**self.static_data, **other.static_data}
        return LazyExecutionGraph(
            steps=other.steps,
            static_data=merged_static,
            input_data=other.input_data,
            input_name=other.input_name,
        )


@dataclass
class LazyExecutionGraph:
    steps: OrderedDict[str, Step]
    static_data: t.Dict[str, t.Any]
    input_data: pl.LazyFrame
    input_name: str

    def compute(self, **kwargs) -> pd.DataFrame:
        """Execute all steps in order, passing data between them."""
        data: t.Dict[str, t.Any] = {**self.static_data, **kwargs}
        final_step = None
        for step_name, step in self.steps.items():
            input_values = {item.param_name: data[item.node_name] for item in step.input_map}
            data[step_name] = step.func(**input_values)
            final_step = step_name
        return data[final_step]

    def merge(self, other: "LazyExecutionGraph|EmptyExecutionGraph") -> "LazyExecutionGraph":
        """Merge two LazyExecutionGraphs, preserving order and deduplicating steps."""
        merged_steps = OrderedDict(self.steps)
        merged_steps.update((k, v) for k, v in other.steps.items() if k not in self.steps)
        merged_static = {**self.static_data, **other.static_data}
        return LazyExecutionGraph(
            steps=merged_steps,
            static_data=merged_static,
            input_data=self.input_data,
            input_name=self.input_name,
        )


# ---------------------------------------------------------------------------
# Wrapping helpers for PED node callables
# ---------------------------------------------------------------------------


def _is_dataframe_type(annotation: t.Any) -> bool:
    """Return True if the annotation is pd.Series or pd.DataFrame."""
    return annotation in (pd.Series, pd.DataFrame)


def _get_input_annotations(func: t.Callable) -> t.Dict[str, t.Any]:
    """Return a mapping of parameter name → type annotation for *func*."""
    import inspect

    sig = inspect.signature(func)
    hints: t.Dict[str, t.Any] = t.get_type_hints(func) if hasattr(func, "__annotations__") else {}
    return {
        name: hints.get(name, t.Any)
        for name, param in sig.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }



def wrap_input_callable(
    node_name: str,
    original_callable: t.Callable,
    external_input_name: str,
    internal_input_name: str = 'input',
) -> t.Callable:
    """Return a new callable that produces a ``LazyExecutionGraph`` from a ``pl.LazyFrame``.

    The produced graph contains a single step for this input node.  The step
    converts the incoming polars batch (``pd.DataFrame``) to the value expected
    by downstream processing nodes.

    Args:
        node_name: The PEDNode's ``namespaced_name`` (used as the step key).
        original_callable: The original node callable (receives ``pl.LazyFrame``
            via ``external_input_name`` and returns ``pd.DataFrame``).
        external_input_name: The external variable name for the LazyFrame input.
    """

    def wrapped_input(**kwargs: t.Any) -> LazyExecutionGraph:
        return LazyExecutionGraph(
            steps=OrderedDict(
                [(node_name, Step(
                    func=original_callable, 
                    input_map=[InputMapItem(node_name=external_input_name, param_name=internal_input_name)]
                ))]
            ),
            static_data={},
            input_data=kwargs[external_input_name],
            input_name=external_input_name,
        )

    wrapped_input.__name__ = node_name
    wrapped_input.__doc__ = f"Lazy input node: {node_name}"
    return wrapped_input


def wrap_processing_callable(
    node_name: str,
    original_callable: t.Callable,
    input_map: t.List[InputMapItem],
) -> t.Callable:
    """Return a callable that accumulates a computation step into a ``LazyExecutionGraph``.

    If none of the incoming kwargs contain a ``LazyExecutionGraph``, the node is
    executed eagerly (static/constant case).

    Args:
        node_name: Step key in the LazyExecutionGraph.
        original_callable: The original node callable.
        input_map: Mapping of external variable names to internal function parameter names.
            Records which inputs the step requires. 
    """

    def wrapped_processing(**kwargs: t.Any) -> t.Any:
        lazy_graph: t.Union[EmptyExecutionGraph, LazyExecutionGraph] = EmptyExecutionGraph()

        for param_name, value in kwargs.items():
            if isinstance(value, LazyExecutionGraph):
                lazy_graph = lazy_graph.merge(value)
            else:
                lazy_graph.static_data[param_name] = value

        if lazy_graph.steps is None:
            # No lazy inputs – execute eagerly
            return original_callable(**kwargs)

        lazy_graph.steps[node_name] = Step(
            func=original_callable,
            input_map=input_map,
        )
        return lazy_graph

    wrapped_processing.__name__ = node_name
    wrapped_processing.__doc__ = f"Lazy processing node: {node_name}"
    return wrapped_processing


def wrap_collector_callable(
    node_name: str,
    original_callable: t.Callable,
    all_input_names: t.List[str],
    schema: "SchemaDict",
    map_batches_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Callable:
    """Return a callable that finalises a ``LazyExecutionGraph`` via ``map_batches``.

    First wraps the node as a processing node (to accumulate the final collection
    step), then attaches the ``map_batches`` call that materialises the result.

    Args:
        node_name: Step key in the LazyExecutionGraph.
        original_callable: The original collector callable.
        all_input_names: All external variable names this node receives.
        schema: Polars output schema passed to ``map_batches``.
        map_batches_kwargs: Additional keyword arguments forwarded to
            ``LazyFrame.map_batches``.
    """
    _map_batches_kwargs = map_batches_kwargs or {}
    processing_callable = wrap_processing_callable(
        node_name, 
        original_callable, 
        input_map=[InputMapItem(node_name=name, param_name=name) for name in all_input_names],
    )

    def wrapped_collector(**kwargs: t.Any) -> t.Any:
        result = processing_callable(**kwargs)
        if not isinstance(result, LazyExecutionGraph):
            return result
        return result.input_data.map_batches(
            lambda df: pl.from_pandas(result.compute(**{result.input_name: df})),
            schema=schema,
            **_map_batches_kwargs,
        )

    wrapped_collector.__name__ = node_name
    wrapped_collector.__doc__ = f"Lazy collector node: {node_name}"
    return wrapped_collector
