from fastapi import APIRouter

router = APIRouter(prefix="/server", tags=["server"])

__all__ = ["router"]
