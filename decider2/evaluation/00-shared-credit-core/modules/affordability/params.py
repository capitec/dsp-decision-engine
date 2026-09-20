"""Parameters for affordability capability."""

from pydantic import BaseModel, Field


class AffordabilityParams(BaseModel):
    """Affordability assessment parameters.

    These are Credit Risk Policy parameters changed on a quarterly basis.
    """

    affordability_buffer: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Buffer applied to discretionary income — max instalment ratio",
    )
    marginal_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Threshold at which verdict moves from pass to marginal",
    )
    min_expense_floor_base: float = Field(
        3000.0, ge=0.0, description="Base minimum living expense floor"
    )
    min_expense_floor_per_dependent: float = Field(
        500.0, ge=0.0, description="Additional floor per dependent"
    )
