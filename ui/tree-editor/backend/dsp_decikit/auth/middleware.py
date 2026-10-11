"""Rolling session middleware for automatic session renewal."""

import time
from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from dsp_decikit.settings import settings
from dsp_decikit.auth.sessions import get_session, renew_session
from dsp_decikit.auth.models import SessionInfo
from logging import getLogger

logger = getLogger(__name__)


class RollingSessionMiddleware(BaseHTTPMiddleware):
    """Middleware to handle automatic rolling session renewal."""

    # Public endpoints that don't require authentication
    PUBLIC_PATHS = {
        "/api/health",  # Health check for monitoring/orchestration
        "/api/auth/config",  # Auth configuration endpoint
        "/api/auth/oidc/login/{provider}",  # OIDC login initiation (matches any provider)
        "/api/auth/oidc/token",  # OIDC redirect callback
        "/api/auth/basic/login",  # Basic auth login
        "/api/auth/basic/register",  # Basic auth registration
    }

    def __init__(self, app):
        super().__init__(app)
        self._logout_url_cache = None

    def _is_public_path(self, path: str) -> bool:
        """Check if the request path is a public endpoint."""
        if path in self.PUBLIC_PATHS:
            return True
        # Handle dynamic paths like /api/auth/oidc/login/{provider}
        if path.startswith("/api/auth/oidc/login/"):
            return True
        return False

    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        # Skip middleware for non-API paths (user won't have session cookies anyway)
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        # Skip middleware for public endpoints (don't require authentication)
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Get session token from cookies
        session_token = request.cookies.get(settings.auth.cookie.key)
        if not session_token:
            return await call_next(request)

        # Skip session renewal for logout endpoint (no point renewing if logging out)
        if request.url.path == self._get_logout_url(request):
            return await call_next(request)

        session_info = await get_session(session_token)
        request.state.session = (session_token, session_info)
        # Unknown sessions go through: the route's own auth refuses them.
        if session_info and int(time.time()) >= session_info.expires_at:
            logger.info(f"Session {session_token[:8]}... expired, redirecting to logout")
            return RedirectResponse(url=self._get_logout_url(request), status_code=302)

        response = await call_next(request)

        if session_info and (
            session_info.expires_at - int(time.time())
            <= settings.auth.session_renewal_threshold_s
        ):
            try:
                await self._renew_session(session_token, session_info, response)
            except Exception as e:
                # The request itself succeeded; failing it now would misreport it.
                logger.error(f"Error renewing session: {e}")

        return response

    def _get_logout_url(self, request: Request) -> str:
        """Get logout URL dynamically using FastAPI's url_path_for with caching."""
        if self._logout_url_cache is None:
            try:
                self._logout_url_cache = request.app.url_path_for("auth_logout")
            except Exception:
                # Fallback to hardcoded path if route name not found
                logger.warning("Could not find auth_logout route, using fallback URL")
                self._logout_url_cache = "/api/auth/logout"

        return self._logout_url_cache

    async def _renew_session(
        self, session_token: str, session_info: SessionInfo, response: Response
    ):
        """Renew the session and update cookies."""
        current_time = int(time.time())

        # Update session info - only last_activity changes (expires_at is calculated dynamically)
        session_info.last_activity = current_time

        await renew_session(session_token, session_info)

        # Update status cookie with new expiry timestamp (calculated from new last_activity)
        new_expires_at = (
            session_info.expires_at
        )  # This uses the property which calculates from last_activity

        response.set_cookie(
            key=settings.auth.status_cookie.key,
            value=str(new_expires_at),
            max_age=settings.auth.session_max_duration_s,
            **settings.auth.status_cookie.model_dump(mode="json", exclude={"key"}),
        )

        logger.debug(f"Session renewed until {new_expires_at}")
