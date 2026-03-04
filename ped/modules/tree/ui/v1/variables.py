"""Variable types for placeholder support in tree nodes."""

import typing as t
from pydantic import BaseModel, Field
from enum import Enum


class VariableType(str, Enum):
    """Supported variable types matching frontend VariableType."""

    STRING = "string"
    NUMERIC = "numeric"
    NUMERIC_EXPR = "numeric_expr"


_VT = t.TypeVar("_VT", bound=VariableType)
_T = t.TypeVar("_T")


class _BasePlaceHolderVariable(BaseModel, t.Generic[_VT, _T]):
    """Represents a placeholder variable reference in tree nodes.

    This matches the frontend VariableInfo structure but simplified for backend use.
    """

    name: str = Field(description="Variable name", pattern="^[a-z][a-z0-9_]*$")
    var_type: _VT = Field(description="Variable type")
    value: _T = Field(description="Variable value")

    def __str__(self) -> str:
        return f"#{self.name}"


class NumericVariable(
    _BasePlaceHolderVariable[t.Literal[VariableType.NUMERIC], t.Union[int, float]]
):
    pass


class StringVariable(_BasePlaceHolderVariable[t.Literal[VariableType.STRING], str]):
    pass


PlaceHolderVariable = t.Annotated[
    t.Union[NumericVariable, StringVariable], Field(discriminator="var_type")
]

# NOTE: The keys in the dict are variable ids which are stored in the nodes,
# Having a separate variable id over a variable name means we can change the name without having to go to all the nodes and update the id
VariableMap = t.Dict[str, PlaceHolderVariable]
