import typing as t
import polars as pl
from ped.modules.core import BaseModule, PEDNode

class ExtractFromStructModule(BaseModule):
    type: t.Literal['extract_from_struct'] = "extract_from_struct"
    field_names: t.List[str]

    def expand_nodes(self) -> t.List[PEDNode]:
        """
        Expand into PEDNodes for extracting fields from a struct column.
        
        Returns:
            List of PEDNodes
        """
        from .impl import extract_struct_field
        # We need to map the internal 'column' parameter to the external parameter 'input'
        # The 'input' parameter is expected to be provided by the user in the config input_mapping
        return [
            PEDNode.from_callable(
                extract_struct_field,
                name=field_name,
                input_map={"column": "input"},
                static_kwargs={"field_name": field_name}
            ) for field_name in self.field_names
        ]
