from fastapi import APIRouter

TAGS = [{"name": "datasources", "description": "Data source discovery and management"}]

router = APIRouter(prefix="/datasources", tags=["datasources"])

__all__ = ["router", "TAGS"]
