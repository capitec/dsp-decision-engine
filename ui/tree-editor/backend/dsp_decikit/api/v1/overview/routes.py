from fastapi import APIRouter

TAGS = [{"name": "Overview", "description": "Dashboard overview and statistics"}]

router = APIRouter(prefix="/overview", tags=["Overview"])

__all__ = ["router", "TAGS"]
