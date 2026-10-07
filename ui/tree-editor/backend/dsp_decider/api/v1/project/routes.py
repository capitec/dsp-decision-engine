from fastapi import APIRouter

TAGS = [{"name": "Projects", "description": "Project management and sharing"}]

router = APIRouter(tags=["Projects"])

__all__ = ["router", "TAGS"]
