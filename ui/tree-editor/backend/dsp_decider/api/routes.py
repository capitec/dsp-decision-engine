from fastapi import APIRouter
from dsp_decider.api import v1
from . import health
from dsp_decider.settings import settings

TAGS = [
    *v1.TAGS,
    *health.TAGS,
    {"name": "auth", "description": "Authentication operations"},
]

router = APIRouter(prefix="/api")
router.include_router(v1.router)
router.include_router(health.router)

# Initialize auth routes
settings.auth.init_provider(router)

__all__ = ["router", "TAGS"]
