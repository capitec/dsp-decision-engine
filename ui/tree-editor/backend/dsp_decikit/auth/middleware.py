"""Rolling session middleware for automatic session renewal."""

import time
from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from dsp_decikit.settings import settings
from dsp_decikit.cache import get_cache, CacheKeys
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

        # Check session BEFORE processing the request
        try:
            redirect_response = await self._check_session_before_request(
                request, session_token
            )
            if redirect_response:
                return redirect_response
        except Exception as e:
            logger.error(f"Error checking session before request: {e}")
            # Continue with request if session check fails

        # Process the actual request
        response = await call_next(request)

        # Handle session renewal AFTER successful request
        try:
            await self._handle_rolling_session(request, session_token, response)
        except Exception as e:
            logger.error(f"Error in rolling session middleware: {e}")
            # Don't break the request if session renewal fails

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

    async def _check_session_before_request(
        self, request: Request, session_token: str
    ) -> StarletteResponse | None:
        """Check if session is expired before processing request."""
        cache = get_cache()
        cache_key = CacheKeys.auth_session(session_token)
        session_data = await cache.get(cache_key, None)

        if not session_data:
            return None  # Invalid session, let auth system handle it

        try:
            session_info = SessionInfo.model_validate(session_data)
        except Exception:
            return None  # Invalid session data, let auth system handle it

        current_time = int(time.time())

        # Check if session has expired (handles both rolling and max duration limits)
        if current_time >= session_info.expires_at:
            logger.info(
                f"Session {session_token[:8]}... expired, redirecting to logout"
            )
            # Session expired - redirect to logout endpoint
            return RedirectResponse(url=self._get_logout_url(request), status_code=302)

        return None  # Session is valid, continue with request

    async def _handle_rolling_session(
        self, request: Request, session_token: str, response: Response
    ):
        """Check session and renew if needed after successful request."""
        logger.debug(
            f"Checking session renewal for {request.url.path} (token: {session_token[:8]}...)"
        )
        cache = get_cache()
        cache_key = CacheKeys.auth_session(session_token)
        session_data = await cache.get(cache_key, None)

        if not session_data:
            return  # Invalid session, let auth system handle it

        try:
            session_info = SessionInfo.model_validate(session_data)
        except Exception:
            return  # Invalid session data, let auth system handle it

        current_time = int(time.time())

        # Session should not be expired here (we checked before request)
        # But check if session needs renewal (within threshold of expiry)
        time_until_expiry = session_info.expires_at - current_time
        if time_until_expiry <= settings.auth.session_renewal_threshold_s:
            logger.info(
                f"Renewing session {session_token[:8]}... (expires in {time_until_expiry}s)"
            )
            await self._renew_session(session_token, session_info, cache, response)

    async def _renew_session(
        self, session_token: str, session_info: SessionInfo, cache, response: Response
    ):
        """Renew the session and update cookies."""
        current_time = int(time.time())

        # Update session info - only last_activity changes (expires_at is calculated dynamically)
        session_info.last_activity = current_time

        # Save updated session to cache with max duration (cache handles expiry based on timestamps)
        cache_key = CacheKeys.auth_session(session_token)
        await cache.put(
            cache_key,
            session_info.model_dump(mode="json"),
            expire_seconds=settings.auth.session_max_duration_s,  # Use max duration for cache TTL
        )

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
