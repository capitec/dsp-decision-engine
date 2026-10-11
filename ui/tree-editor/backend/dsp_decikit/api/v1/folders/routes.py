from fastapi import APIRouter

TAGS = [{"name": "folders", "description": "Folder management and organization"}]

router = APIRouter(prefix="/folders", tags=["folders"])

__all__ = ["router", "TAGS"]
