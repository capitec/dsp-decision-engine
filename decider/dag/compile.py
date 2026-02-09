import typing as t
from types import ModuleType
import inspect
import polars as pl
from polars.expr import Expr
from typing import Any, Dict, Optional
from hamilton import lifecycle
from polars._typing import SchemaDefinition

from dataclasses import dataclass
from hamilton import node
from .expanders.base import DeciderExpandableModule
from hamilton.driver import Builder

from .expanders.lazyframe import LazyFrameModule

if t.TYPE_CHECKING:
    from hamilton import node


class PolarsAliasAdaptor(lifecycle.NodeExecutionMethod):
    """Adaptor that automatically adds .alias(node_name) to Polars Expr results"""
    
    def run_to_execute_node(
        self,
        *,
        node_name: str,
        node_tags: Dict[str, Any],
        node_callable: Any,
        node_kwargs: Dict[str, Any],
        task_id: Optional[str],
        is_expand: bool,
        is_collect: bool,
        **future_kwargs: Any,
    ) -> Any:
        # Execute the original node
        result = node_callable(**node_kwargs)
        
        # Check if this is a Polars Expr return type # TODO check i can rather just use node_kwargs
        sig = inspect.signature(node_callable)
        if hasattr(sig, 'return_annotation') and 'Expr' in str(sig.return_annotation):
            # Add .alias(node_name) for Polars expressions
            return result.alias(node_name)
        
        # Return original result for non-Expr types
        return result
    

class LazyFrameBuilder(lifecycle.ResultBuilder):
    def __init__(self, input_schema: SchemaDefinition):
        """Add a constructor to accept a base LazyFrame"""
        self.input_schema = input_schema
    
    def build_result(self, **outputs: Dict[str, Expr]) -> pl.LazyFrame:
        """Custom function that combines Polars expressions into a LazyFrame"""
        expressions = list(outputs.values())
        return pl.LazyFrame(schema=self.input_schema).with_columns(expressions).select(*outputs.keys())
    

@dataclass
class CompiledModule(DeciderExpandableModule):
    module: ModuleType
    input_schema: SchemaDefinition
    output_vars: t.List[str]

    def _compile(self):
        dr = (
            Builder()
            .with_config({})
            .with_modules(self.module)
            .with_adapters(PolarsAliasAdaptor(),LazyFrameBuilder(self.input_schema))
            .build()
        )
        input_data = {n: pl.col(n) for n in self.input_schema.keys()}
        return dr.execute(final_vars=self.output_vars, inputs=input_data)

    def expand_nodes(self):
        # return LazyFrameModule(self._compile, "TODO").expand_nodes()
        pass
    

@dataclass
class CompiledModulePlaceholder(DeciderExpandableModule):
    module: ModuleType


    def expand_nodes(self):
        # return LazyFrameModule(self._compile, "TODO").expand_nodes()
        # This is just a placeholder for now:
        def process_plan() -> pl.LazyFrame:
            return 1 # In future it will be the lazy frame to execute
        def output(process_plan: pl.LazyFrame, inputs: pl.DataFrame) -> pl.LazyFrame:
            return 1 # In future it will be the lazy frame to execute

        return {
            "process_plan": node.Node.from_fn(process_plan, name="process_plan"),
            "output": node.Node.from_fn(output, name="output")
        }