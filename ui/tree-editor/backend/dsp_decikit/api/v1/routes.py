from fastapi import APIRouter
from . import project, user, teams, auth, datasources, overview, folders, server

TAGS = [
    *project.TAGS,
    *user.TAGS,
    *teams.TAGS,
    *auth.TAGS,
    *datasources.TAGS,
    *overview.TAGS,
    *folders.TAGS,
    *server.TAGS,
]

router = APIRouter(prefix="/v1")
router.include_router(project.router)
router.include_router(user.user_router)
router.include_router(user.users_router)
router.include_router(teams.router)
router.include_router(auth.router)
router.include_router(datasources.router)
router.include_router(overview.router)
router.include_router(folders.router)
router.include_router(server.router)

__all__ = ["router", "TAGS"]
