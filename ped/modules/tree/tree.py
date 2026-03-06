import typing as t
from pydantic import Field
from .nodes import NodeType
from ped.serializable.dataframe import TDataFrameData, TDataFrameRow, PolarsSchema, build_polars_df
from .shared import WithTreeOutput

class Tree(WithTreeOutput):
    root: NodeType
    parameters_col: str = "parameters"
    default_parameters: t.Dict[str, t.Any] = Field(default_factory=dict)

    def get_required_features(self) -> t.Set[str]:
        return self.root.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.root.get_required_parameters()


# class TreeBuilder:
#     def with_output_df(dataframe: pl.DataFrame):
#         ...
#     def with_default(...):
#         ...
# TODO RangePattern.branch(min=10,max=20).then(5).otherwise(10)
# or RangePattern
#    .branch(min=10,max=20).then(StringPattern).branch('asdf').then(0)
#    .branch(min=20).then(1).otherwise(2)
# Then either pushes to the stack if the value is a class or pops from the stack if the output is an index

# Or maybe we can even be smart and say TreeBuilder.with_polars_expression(expr: pl.Expr)
# expr = pl.when(x=10).then(...) and we have certain checks we can port over.