import typing as t
import polars as pl
from ..core import BaseModule
from ..util import create_node_with_mapping

class ExtractFromStructModule(BaseModule):
    type: t.Literal['extract_from_struct'] = "extract_from_struct"
    field_names: t.List[str]

    def create_nodes(self) -> t.Dict[str, pl.Expr]:
        """
        Create Hamilton nodes for extracting fields from a struct column.
        
        Returns:
            Dict mapping output node names to Hamilton nodes
        """
        from .impl import extract_struct_field
        return [
            create_node_with_mapping(
                extract_struct_field,
                name=field_name,
                input_mapping={"input": "column"},
                partial_kwargs={"field_name": field_name}
            ) for field_name in self.field_names
        ]