import ast
import types
import typing as t
from pydantic import BaseModel, Field, PrivateAttr, model_validator, RootModel
import polars as pl


def extract_names_and_parameters(expression: str) -> t.Tuple[t.Set[str], t.Set[str]]:
    """
    Extract variable names and parameter accesses from a Python expression.

    Returns:
        (variable_names, parameter_names) where:
        - variable_names: set of standalone names like 'b', 'age' (excludes function names)
        - parameter_names: set of parameter accesses like 'p.asdf', 'params.income'
    """
    try:
        tree = ast.parse(expression, mode="eval")
        all_names = []
        parameter_names = set()
        exclude_name_ids = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_names.append(node)
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "p"
            ):
                param_name = node.attr
                if param_name:
                    parameter_names.add(param_name)
                    exclude_name_ids.add(id(node.value))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    exclude_name_ids.add(id(node.func))

        variable_names = {
            name.id for name in all_names if id(name) not in exclude_name_ids
        }

        return variable_names, parameter_names

    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e


ALLOWED_POLARS_FUNCTIONS = {
    "duration": pl.duration,
    "datetime": pl.datetime,
    "date": pl.date,
    "time": pl.time,
    "lit": pl.lit,
    # ... To Be Extended with more allowed functions as needed
}


class _ComputedFeature(BaseModel):
    """Definition of a computed feature based on existing features."""

    type: t.Literal["computed"] = "computed"
    expression: str = Field(
        description="Expression to compute the feature (e.g., 'feature1 + feature2')"
    )
    _features: t.Set[str] = PrivateAttr(default_factory=set)
    _parameters: t.Set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def extract_features_and_parameters(self):
        features, parameters = extract_names_and_parameters(self.expression)
        self._features = features
        self._parameters = parameters
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Dict[str, pl.Expr],
    ) -> pl.Expr:
        try:
            from simpleeval import simple_eval
        except ImportError:
            raise ImportError(
                "simpleeval is required to evaluate computed feature expressions. Please install it with 'pip install simpleeval'."
            )

        p = types.SimpleNamespace(
            **{name: parameters.struct.field(name) for name in self._parameters}
        ) if parameters is not None else None
        res = simple_eval(
            self.expression,
            names={**inputs, "p": p},
            functions=ALLOWED_POLARS_FUNCTIONS,
        )
        assert isinstance(
            res, pl.Expr
        ), f"Expression {self.expression} did not evaluate to a Polars expression: {res}.\nHint: if the intended output is as expected you may want to consider wrapping it in lit()"
        return res


class Feature(RootModel[t.Union[_ComputedFeature, str]]):
    root: t.Union[_ComputedFeature, str] = Field(description="Feature name to test")

    def __str__(self) -> str:
        """String representation returns the feature name or expression."""
        if isinstance(self.root, str):
            return self.root
        return self.root.expression

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Dict[str, pl.Expr],
    ) -> pl.Expr:
        if isinstance(self.root, str):
            return inputs[self.root]
        return self.root.build_expression(inputs, parameters)

    def get_required_features(self) -> t.Set[str]:
        if isinstance(self.root, str):
            return {self.root}
        return self.root._features

    def get_required_parameters(self) -> t.Set[str]:
        if isinstance(self.root, str):
            return set()
        return self.root._parameters
