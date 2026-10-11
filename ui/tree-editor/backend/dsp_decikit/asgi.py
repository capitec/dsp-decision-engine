from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from . import api
from .services.documents import add_document_schemas
from .settings import settings
from .spa_static_files import SPAStaticFiles
from .exceptions import register_handlers
from .auth.middleware import RollingSessionMiddleware
from .services.live_hub import hub


@asynccontextmanager
async def lifespan(_app):
    if settings.live_collab:
        hub.start()
    yield
    await hub.stop()


app = FastAPI(
    title="Project Version Management",
    openapi_tags=api.TAGS,
    docs_url="/swagger",
    lifespan=lifespan,
)

register_handlers(app)

_default_openapi = app.openapi
app.openapi = lambda: add_document_schemas(_default_openapi())  # idempotent

# Add rolling session middleware
app.add_middleware(RollingSessionMiddleware)

# Include API routes first (they take precedence)
app.router.include_router(router=api.router)

# Set up static file serving for React build if configured
if settings.ui_build_path:
    ui_build_path = Path(settings.ui_build_path)
    if ui_build_path.exists() and ui_build_path.is_dir():
        # Mount using SPAStaticFiles for proper SPA routing
        # API routes already mounted above will take precedence
        app.mount("/", SPAStaticFiles(directory=str(ui_build_path)), name="ui")

# TODO Please initialise auth with the api.router
