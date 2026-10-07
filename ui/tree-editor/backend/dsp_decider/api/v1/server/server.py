from fastapi import Depends
from pydantic import BaseModel
import typing as t

from dsp_decider.auth.authorization import get_current_user_session
from dsp_decider.auth.models import SessionInfo
from .routes import router

try:
    import decider

    DECIDER_VERSION = decider.__version__
except (ImportError, AttributeError):
    DECIDER_VERSION = "unknown"


class ServerInfoResponse(BaseModel):
    """Server information including installed package versions."""

    decider_version: str
    service: str = "dsp-decider"


@router.get("/info", response_model=ServerInfoResponse)
async def get_server_info(
    _session: SessionInfo = Depends(get_current_user_session),
) -> ServerInfoResponse:
    """
    Get server information including installed decider version.

    Requires authentication.
    """
    return ServerInfoResponse(decider_version=DECIDER_VERSION)


__all__ = ["ServerInfoResponse"]
