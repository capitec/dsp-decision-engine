import typing as t
from typing import List
from functools import lru_cache
import hashlib
import json
import secrets
import time
from pydantic import BaseModel
from fastapi import APIRouter, Response, Depends, status as http_status
from fastapi.exceptions import HTTPException
from fastapi.responses import RedirectResponse
from dsp_decikit.settings import settings
from dsp_decikit.cache import get_cache, CacheProtocol, CacheKeys
from .authorization import session_token_auth

router = APIRouter()


class AuthMethodConfig(BaseModel):
    type: str
    enabled: bool
    display_name: str
    icon_name: str
    registration_enabled: bool = False
    provider_key: t.Optional[str] = (
        None  # For OIDC providers, identifies which provider
    )


class AuthConfigResponse(BaseModel):
    methods: List[AuthMethodConfig]


@lru_cache(maxsize=1)
def _get_auth_config() -> AuthConfigResponse:
    """Compute auth config once and cache it"""
    methods = []

    # Check OIDC providers
    if settings.auth.oidc.enabled:
        for provider_key, provider_config in settings.auth.oidc.providers.items():
            methods.append(
                AuthMethodConfig(
                    type="oidc",
                    enabled=True,
                    display_name=provider_config.get_display_name(),
                    icon_name=provider_config.get_icon_name(),
                    provider_key=provider_key,
                )
            )

    # Check basic auth provider
    if settings.auth.basic.enabled:
        methods.append(
            AuthMethodConfig(
                type="basic",
                enabled=True,
                display_name="Username & Password",
                icon_name="key",
                registration_enabled=settings.auth.basic.registration_enabled,
            )
        )

    return AuthConfigResponse(methods=methods)


@lru_cache(maxsize=1)
def _get_config_etag() -> str:
    """Generate ETag based on the config content"""
    config = _get_auth_config()
    config_json = config.model_dump_json()
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


@router.get("/config", response_model=AuthConfigResponse)
async def get_auth_config(response: Response):
    """Get available authentication methods and their configuration"""
    # Set cache headers for client-side caching (1 hour)
    response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers["ETag"] = f'"{_get_config_etag()}"'

    config = _get_auth_config()
    print(f"Auth config being sent: {config.model_dump_json(indent=2)}")
    return config


@router.get("/logout", name="auth_logout")
async def logout(
    session_token: str | None = Depends(session_token_auth),
    cache: CacheProtocol = Depends(get_cache),
):
    """Logout user by clearing their session from cache and all cookies"""
    # Clear session from cache if it exists (ignore if already expired/deleted)
    if session_token:
        cache_key = CacheKeys.auth_session(session_token)
        try:
            await cache.delete(cache_key)
        except KeyError:
            # Session may have already expired or been deleted, which is fine
            pass

    # Create redirect response to login page
    response = RedirectResponse(url="/login", status_code=302)

    # Clear all auth-related cookies
    # Clear main session cookie
    response.delete_cookie(
        **settings.auth.cookie.model_dump(mode="json", exclude={"key"}),
        key=settings.auth.cookie.key,
    )

    # Clear session status cookie
    response.delete_cookie(
        **settings.auth.status_cookie.model_dump(mode="json", exclude={"key"}),
        key=settings.auth.status_cookie.key,
    )

    # Clear OIDC login session cookie if OIDC is enabled
    if settings.auth.oidc.enabled:
        response.delete_cookie(
            **settings.auth.oidc.cookie.model_dump(mode="json", exclude={"key"}),
            key=settings.auth.oidc.cookie.key,
        )

    return response
