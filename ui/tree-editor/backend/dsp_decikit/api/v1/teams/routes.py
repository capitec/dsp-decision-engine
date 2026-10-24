from fastapi import APIRouter

TAGS = [
    {
        "name": "Team Management",
        "description": "Team administration and membership management",
    }
]

router = APIRouter(prefix="/teams", tags=["Team Management"])

__all__ = ["router", "TAGS"]
