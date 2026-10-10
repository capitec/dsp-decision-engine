from fastapi import APIRouter

TAGS = [
    {
        "name": "API Tokens",
        "description": "API token management for programmatic access",
    }
]

router = APIRouter(prefix="/auth/tokens", tags=["API Tokens"])

__all__ = ["router", "TAGS"]
