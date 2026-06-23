import typing as t
from pydantic import model_validator
from decider.modules.expression import ExpressionModule, Node, ExternalInputNode
from .config import ParametersConfig, Expression



class DecisionTableModule(ExpressionModule):
    type: t.Literal["decision_table"]
    parameters: ParametersConfig
    expression: Expression
    outputs: t.List[str]
    default: t.Optional[t.List[t.Any]] = None
    
    @model_validator(mode='after')
    def validate_config(self):
        # Validate that all output columns exist in parameters
        for output in self.outputs:
            if output not in self.parameters.columns:
                raise ValueError(f"Output column '{output}' not found in parameters columns")
        
        # Validate default has correct length if specified
        if self.default is not None:
            if len(self.default) != len(self.outputs):
                raise ValueError(f"Default values length ({len(self.default)}) must match outputs length ({len(self.outputs)})")
        
        # Validate expression parameters
        self.expression.validate_parameters(self.parameters)
        
        return self
    
    def expand_nodes(self) -> t.Dict[str, Node]:
        from .impl import calculate_decision_table_output

        variables = self.expression.get_variables()
        input_map = {v: v for v in variables}

        node = Node.from_callable(
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
        return {node.name: node}
