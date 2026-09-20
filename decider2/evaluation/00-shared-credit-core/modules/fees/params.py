"""Parameters for fee calculation capability."""

from pydantic import BaseModel, Field


class FeesParams(BaseModel):
    """Fee calculation parameters.

    These are library-global parameters owned by Treasury and Compliance.
    """

    initiation_fee_rate: float = Field(
        0.015, ge=0.0, le=0.1, description="Initiation fee as % of offered amount"
    )
    initiation_fee_cap: float = Field(
        2000.0, ge=0.0, description="Statutory cap on initiation fee"
    )
    monthly_service_fee_rate: float = Field(
        0.005, ge=0.0, le=0.1, description="Monthly service fee as % of offered amount"
    )
    monthly_service_fee_cap: float = Field(
        150.0, ge=0.0, description="Statutory cap on monthly service fee"
    )
    base_premium_rate: float = Field(
        0.5, ge=0.0, description="Base credit life premium per R1000 of cover"
    )
    max_age_for_cover: float = Field(
        75.0, ge=0.0, description="Maximum age for credit life insurance"
    )
    youth_age_multiplier: float = Field(
        0.8, ge=0.0, le=2.0, description="Premium multiplier for ages under 25"
    )
    senior_age_multiplier: float = Field(
        1.5, ge=0.0, le=2.0, description="Premium multiplier for ages over 60"
    )
