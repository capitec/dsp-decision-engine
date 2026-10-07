from fastapi import APIRouter
from pydantic import BaseModel
import typing as t

router = APIRouter(tags=["health"])
TAGS = [
    {"name": "health", "description": "Health check operations"},
]


class HealthResponse(BaseModel):
    status: t.Literal["ok"] = "ok"
    service: str = "dsp-decider"


# Probably petty but stops us needing to construct it each time
health_response = HealthResponse()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration and monitoring."""
    return health_response


__all__ = ["router", "TAGS"]
