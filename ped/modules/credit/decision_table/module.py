import typing as t
from pydantic import model_validator
from ped.modules.core import BaseModule, PEDNode
from .config import Expression
from ped.serializable.dataframe import DataFrame


class DecisionTableModule(BaseModule):
    type: t.Literal["decision_table"]
    parameters: DataFrame
    expression: Expression
    outputs: t.List[str]
    default: t.Optional[t.List[t.Any]] = None
    
    @model_validator(mode='after')
    def validate_config(self):
        # Validate that all output columns exist in parameters
        for output in self.outputs:
            if output not in self.parameters.df.columns:
                raise ValueError(f"Output column '{output}' not found in parameters columns")
        
        # Validate default has correct length if specified
        if self.default is not None:
            if len(self.default) != len(self.outputs):
                raise ValueError(f"Default values length ({len(self.default)}) must match outputs length ({len(self.outputs)})")
        
        # Validate expression parameters
        self.expression.validate_parameters(self.parameters)
        
        return self
    
    def expand_nodes(self, config: t.Dict[str, t.Any] = None) -> t.List[PEDNode]: # noqa: ARG002
        """
        Expand the decision table configuration into PEDNodes.
        
        Returns:
            List of PEDNodes
        """
        from .impl import calculate_decision_table_output
        
        variables = self.expression.get_variables()
        input_map = {v: v for v in variables}

        # Create the main decision table evaluation node
        return [
            PEDNode.from_callable(
                calculate_decision_table_output,
                name="output",
                input_map=input_map,
                static_kwargs={
                    "parameters": self.parameters.df,
                    "expression": self.expression,
                    "output_columns": self.outputs,
                    "default": self.default
                }
            )
        ]
