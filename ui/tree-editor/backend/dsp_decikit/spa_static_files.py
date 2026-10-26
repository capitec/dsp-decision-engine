import os
from pathlib import Path
from typing import Any, Dict, Optional

from starlette.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.types import Scope
from logging import getLogger

logger = getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    """
    Custom StaticFiles that serves index.html for any unmatched routes,
    enabling proper Single Page Application (SPA) routing behavior.
    """

    def __init__(self, *, directory: str | Path, index: str = "index.html") -> None:
        super().__init__(directory=directory, html=True)
        self.index = index
        self.index_path = Path(directory) / index

        # Pre-create the FileResponse for index.html if it exists
        self._index_response: Optional[FileResponse] = None
        if self.index_path.exists():
            # Pre-compute headers with ETag based on file modification time
            mtime = self.index_path.stat().st_mtime
            headers = {
                "Cache-Control": "public, max-age=3600, must-revalidate",
                "ETag": f'"{mtime}"',
            }
            self._index_response = FileResponse(
                self.index_path,
                media_type="text/html",
                headers=headers,
            )

    async def get_response(self, path: str, scope: Scope) -> FileResponse:
        """
        Serve files from the static directory, falling back to index.html
        for any paths that don't match existing files (SPA routing).
        """
        try:
            # Try to serve the requested file normally
            return await super().get_response(path, scope)
        except HTTPException as exc:
            # If file not found (404), serve pre-created index.html response for SPA routing
            if (
                exc.status_code == 404
                and self._index_response is not None
                and not path.startswith("/api/")
                and not path.startswith("api/")
            ):
                logger.info(f"path {path} not found redirecting to index.html")
                return self._index_response
            # Re-raise other exceptions
            raise exc
