from fastapi import Depends
from pydantic import BaseModel
import typing as t

from dsp_decikit.auth.authorization import get_current_user_session
from dsp_decikit.auth.models import SessionInfo
from .routes import router

try:
    import decikit

    DECIKIT_VERSION = decikit.__version__
except (ImportError, AttributeError):
    DECIKIT_VERSION = "unknown"


class ServerInfoResponse(BaseModel):
    """Server information including installed package versions."""

    decikit_version: str
    service: str = "dsp-decikit"


@router.get("/info", response_model=ServerInfoResponse)
async def get_server_info(
    _session: SessionInfo = Depends(get_current_user_session),
) -> ServerInfoResponse:
    """
    Get server information including installed decikit version.

    Requires authentication.
    """
    return ServerInfoResponse(decikit_version=DECIKIT_VERSION)


__all__ = ["ServerInfoResponse"]
