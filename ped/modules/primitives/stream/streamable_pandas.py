"""StreamablePandasModule – wraps PED modules for polars map_batches execution.

Inherits from :class:`~ped.modules.primitives.namespaced.NamespacedModule` so it
accepts the same ``modules`` list and follows the same namespace/expand pattern.
Rather than explicit ``external_input_name``/``internal_input_name`` fields the
module reuses the ``input_name`` and ``output_name`` fields from
:class:`~ped.modules.core.BaseModule`:

* ``input_name`` — the **external** parameter name (the ``pl.LazyFrame`` coming in).
* The **internal** name (the ``pd.DataFrame`` seen by the inner module's functions)
  is derived from the module's own namespace, e.g. ``"<module_name>.<input_name>"``.
  Override :meth:`_internal_input_name` to customise this derivation.
* ``output_name`` — the name of the injected collector output node.

Usage example::

    from ped.modules.primitives import HamiltonModule
    from ped.modules.primitives.stream import StreamablePandasModule

    inner = HamiltonModule(
        name="transforms",
        module_paths=["my_package.transforms"],
        base_import_path="/path/to/package",
    )

    stream = StreamablePandasModule(
        name="stream_transforms",
        modules=[inner],
        output_columns=["col_a", "col_b"],
        output_schema={"col_a": pl.Int64, "col_b": pl.Float64},
        # input_name defaults to "input", output_name defaults to "output"
    )
"""
import typing as t
from dataclasses import dataclass

import pandas as pd
import polars as pl
from pydantic import Field, model_validator

from ped.modules.core import PEDNode
from ped.modules.primitives.namespaced import NamespacedModule
from ped.serializable.schema import PolarsSchema

from .lazy_execution import (
    _is_dataframe_type,
    _get_input_annotations,
    wrap_input_callable,
    wrap_processing_callable,
    wrap_collector_callable,
    InputMapItem,
)


# ---------------------------------------------------------------------------
# Callable objects for the injected nodes
# ---------------------------------------------------------------------------




def input_converter(input: t.Union[pl.LazyFrame, pl.DataFrame]) -> pd.DataFrame:
    """This is needed as the Polars lazyframe stream will batch data into smaller polars dataframes not pandas dataframes"""
    
    if isinstance(input, pd.DataFrame):
        return input
    return input.to_pandas() # I think it will always be a pl.dataframe here not sure if i should collect if its lazy

@dataclass
class OutputExtractor:
    """Callable object that collects named columns into a single output ``pd.DataFrame``.

    Column order is fixed to ``output_columns`` so the polars output schema is
    applied deterministically.

    Attributes:
        output_columns: Ordered list of column names to collect.
    """

    output_columns: t.List[str]

    def __call__(self, **kwargs: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({col: kwargs[col] for col in self.output_columns})


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class StreamablePandasModule(NamespacedModule):
    """Wraps a list of PED modules and makes them executable via polars ``map_batches``.

    Inherits from :class:`~ped.modules.primitives.namespaced.NamespacedModule` so
    the ``modules`` field and namespace/promote logic are reused directly.

    On top of the standard namespaced expansion two synthetic PEDNodes are
    injected:

    1. An **input node** (namespaced name = ``_internal_input_name()``) that
       converts the incoming ``pl.LazyFrame`` to a ``pd.DataFrame`` and creates a
       :class:`~.lazy_execution.LazyExecutionGraph`.
    2. A **collector node** (name = :attr:`output_name`) that gathers
       :attr:`output_columns` into a ``pd.DataFrame`` and triggers the final
       ``map_batches`` call.

    Every PEDNode that has ``pd.Series`` or ``pd.DataFrame``-typed inputs is
    wrapped for lazy execution; all other nodes pass through unchanged (they
    execute eagerly as static/scalar computations).

    The **internal** input name — the name by which the inner module's functions
    reference the ``pd.DataFrame`` input — is derived via
    :meth:`_internal_input_name`.  By default this is
    ``"<first_module_name>.<input_name>"``, matching how the inner module's node
    is namespaced after
    :meth:`~ped.modules.core.BaseModule.module_namespaced_nodes` is applied.
    Override the method to customise the derivation.

    Attributes:
        output_columns: Ordered list of column names to collect in the output.
        output_schema: Polars schema dict for the output ``LazyFrame``.
        map_batches_kwargs: Extra kwargs forwarded to ``LazyFrame.map_batches``.
    """

    type: t.Literal["streamable_pandas"]  # type: ignore[assignment]

    output_schema: t.Union[PolarsSchema, t.Dict[str, t.Any]] = Field(
        description=(
            "Polars output schema for the collector node. "
            "Accepts either a serialisable ``PolarsSchema`` object or a plain "
            "``{column_name: polars_dtype}`` dict. "
            "The top-level keys determine which columns are collected."
        )
    )
    map_batches_kwargs: t.Dict[str, t.Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to polars LazyFrame.map_batches.",
    )

    # ------------------------------------------------------------------
    # Derived helper: extract column names from whatever schema form was given
    # ------------------------------------------------------------------

    def _output_columns(self) -> t.List[str]:
        """Return ordered column names derived from ``output_schema``."""
        if isinstance(self.output_schema, PolarsSchema):
            return list(self.output_schema.schema.keys())
        return list(self.output_schema.keys())

    def _polars_schema(self) -> t.Any:
        """Return the raw schema value accepted by ``map_batches``."""
        if isinstance(self.output_schema, PolarsSchema):
            return self.output_schema.schema
        return self.output_schema

    @model_validator(mode="after")
    def _validate_output_name_free(self) -> "StreamablePandasModule":
        """Validate output_name is free and all output columns exist as inner nodes."""
        if self.output_name is None:
            raise ValueError("StreamablePandasModule requires output_name to be set.")
        if self.input_name is None:
            raise ValueError("StreamablePandasModule requires input_name to be set.")

        inner_nodes: t.List[PEDNode] = []
        for module in self.modules:
            inner_nodes.extend(module.root.module_namespaced_nodes())

        inner_names = {n.name for n in inner_nodes} | {n.namespaced_name for n in inner_nodes}

        if self.output_name in inner_names:
            raise ValueError(
                f"output_name '{self.output_name}' already exists in the inner modules. "
                "Choose a different name."
            )

        # Verify every output column is resolvable as a node in the inner graph
        output_columns = self._output_columns()
        missing = [col for col in output_columns if col not in inner_names]
        if missing:
            raise ValueError(
                f"The following output_schema columns do not exist as nodes in the "
                f"inner modules: {missing}. "
                f"Available node names: {sorted(inner_names)}"
            )

        return self

    # ------------------------------------------------------------------
    # Internal name derivation
    # ------------------------------------------------------------------

    def _internal_input_name(self) -> str:
        """Return the namespaced name by which inner functions reference the input DataFrame.

        By default this mirrors how the inner module namespaces its nodes:
        ``"<first_module_name>.<input_name>"``.

        Override this method to customise the derivation, for example if the
        inner module applies additional namespace levels.
        """
        if not self.modules:
            raise ValueError("StreamablePandasModule requires at least one inner module.")
        first_module_name = self.modules[0].root.name
        return PEDNode.calculate_namespaced_name(first_module_name, self.input_name)

    # ------------------------------------------------------------------
    # Synthetic node factories
    # ------------------------------------------------------------------

    def _create_input_node(self) -> PEDNode:
        """Inject the input node that converts ``pl.LazyFrame`` → ``pd.DataFrame``."""


        return PEDNode(
            name=self.input_name,
            callable=input_converter,
            original_callable=input_converter,
            namespace=(self.name,),
            input_map={'input': self.input_name},
            extra={"module": "streamable_pandas", "role": "input"},
        )

    def _create_output_node(self) -> PEDNode:
        """Inject the collector node that gathers output columns into a ``pd.DataFrame``."""
        output_columns = self._output_columns()
        extractor = OutputExtractor(output_columns=output_columns)

        return PEDNode(
            name=self.output_name,
            callable=extractor,
            original_callable=extractor.__call__,
            input_map={col: col for col in output_columns},
            extra={"module": "streamable_pandas", "role": "output"},
        )

    # ------------------------------------------------------------------
    # Wrapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_has_dataframe_input(node: PEDNode) -> bool:
        """Return True if any input of *node*'s original callable is ``pd.Series``/``pd.DataFrame``."""
        try:
            annotations = _get_input_annotations(node.type_callable)
        except Exception:
            return False
        return any(_is_dataframe_type(ann) for ann in annotations.values())

    # ------------------------------------------------------------------
    # Core expand_nodes
    # ------------------------------------------------------------------

    def expand_nodes(self) -> t.List[PEDNode]:
        """Expand inner modules, inject synthetic nodes, and wrap dataframe nodes.

        Steps:

        1. Call the parent :class:`NamespacedModule` ``expand_nodes()`` to get
           all inner nodes with child-level namespacing already applied.
        2. Inject the *input* node (name derived from :meth:`_internal_input_name`).
        3. Inject the *collector* output node.
        4. For each node:

           * If it is the input node → wrap with
             :func:`~.lazy_execution.wrap_input_callable`.
           * If it is the collector node → wrap with
             :func:`~.lazy_execution.wrap_collector_callable`.
           * If any of its original-callable inputs are ``pd.Series``/
             ``pd.DataFrame`` → wrap with
             :func:`~.lazy_execution.wrap_processing_callable`.
           * Otherwise → keep as-is (eager).

        Returns:
            Combined list of wrapped PEDNodes.
        """
        # 1. Inner nodes (child-level namespaced via NamespacedModule)
        inner_nodes: t.List[PEDNode] = super().expand_nodes()

        # 2 & 3. Synthetic nodes
        input_node = self._create_input_node()
        output_node = self._create_output_node()

        # Build a lookup keyed by namespaced_name; synthetic nodes win on clash
        all_nodes: t.Dict[str, PEDNode] = {n.namespaced_name: n for n in inner_nodes}
        all_nodes[input_node.namespaced_name] = input_node
        all_nodes[output_node.namespaced_name] = output_node

        internal_input_ns_name = input_node.namespaced_name
        output_ns_name = output_node.namespaced_name

        # 4. Wrap
        wrapped: t.List[PEDNode] = []
        for node in all_nodes.values():
            if node.namespaced_name == internal_input_ns_name:
                new_callable = wrap_input_callable(
                    node_name=node.namespaced_name,
                    original_callable=node.callable,
                    external_input_name=self.input_name,
                    internal_input_name="input", # This is the name the function uses as a parameter so it can be hardcoded
                )
                wrapped.append(
                    PEDNode(
                        name=node.name,
                        callable=new_callable,
                        original_callable=node.original_callable,
                        namespace=node.namespace,
                        input_map={self.input_name: self.input_name},
                        static_kwargs=node.static_kwargs,
                        extra=node.extra,
                    )
                )

            elif node.namespaced_name == output_ns_name:
                new_callable = wrap_collector_callable(
                    node_name=node.namespaced_name,
                    original_callable=node.callable,
                    all_input_names=list(node.input_map.values()),
                    schema=self._polars_schema(),
                    map_batches_kwargs=self.map_batches_kwargs,
                )
                wrapped.append(
                    PEDNode(
                        name=node.name,
                        callable=new_callable,
                        original_callable=node.original_callable,
                        namespace=node.namespace,
                        input_map=node.input_map,
                        static_kwargs=node.static_kwargs,
                        extra=node.extra,
                    )
                )

            elif self._node_has_dataframe_input(node):
                new_input_map = dict(
                    (v_int, internal_input_ns_name) if v_ext == self.input_name else (v_int, v_ext)
                    for v_int, v_ext in node.input_map.items()
                )
                new_callable = wrap_processing_callable(
                    node_name=node.namespaced_name,
                    original_callable=node.callable,
                    input_map=[
                        InputMapItem(node_name=v_ext, param_name=v_int) 
                        for v_int, v_ext in new_input_map.items()
                    ],
                )
                wrapped.append(
                    PEDNode(
                        name=node.name,
                        callable=new_callable,
                        original_callable=node.original_callable,
                        namespace=node.namespace,
                        input_map=new_input_map,
                        static_kwargs=node.static_kwargs,
                        extra=node.extra,
                    )
                )

            else:
                wrapped.append(node)

        return wrapped
