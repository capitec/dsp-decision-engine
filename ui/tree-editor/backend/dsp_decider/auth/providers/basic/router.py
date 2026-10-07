import typing as t
import hashlib
import secrets
import time
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Depends, status as http_status, Form
from fastapi.responses import RedirectResponse
from dsp_decider.settings import settings
from dsp_decider.cache import get_cache, CacheProtocol
from dsp_decider.exceptions import UnauthorizedException
from ...models import UserInfo
from ...authorization import (
    update_user_and_get_session,
    update_user_and_get_session_with_team_roles,
)

router = APIRouter()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    return f"{salt}:{hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        salt, stored_hash = hashed.split(":")
        return (
            stored_hash
            == hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 100000
            ).hex()
        )
    except ValueError:
        return False


@router.post("/login")
async def basic_login(
    email: str = Form(...),
    password: str = Form(...),
    cache: CacheProtocol = Depends(get_cache),
):
    """Login with email/password"""
    # Find user in database
    from ...authorization import find_user

    user = await find_user(email)
    if not user or not user.password_hash:
        raise UnauthorizedException("Invalid credentials")

    if not verify_password(password, user.password_hash.get_secret_value()):
        raise UnauthorizedException("Invalid credentials")

    # Get user teams (already handled in update_user_and_get_session)
    session_id, session_info = await update_user_and_get_session(user, cache)

    response = RedirectResponse(url="/", status_code=302)

    # Set main session cookie (httponly, secure)
    response.set_cookie(
        key=settings.auth.cookie.key,
        value=session_id,
        max_age=settings.auth.session_max_duration_s,  # I tried setting this to rolling duration but it seems this causes it to not allow renew.
        **settings.auth.cookie.model_dump(mode="json", exclude={"key"}),
    )

    # Set session status cookie (JS readable, contains expiry timestamp)
    # Status cookie value should contain the actual session expiry (rolling), not max duration
    expiry_timestamp = int(time.time()) + settings.auth.session_rolling_duration_s
    response.set_cookie(
        key=settings.auth.status_cookie.key,
        value=str(expiry_timestamp),
        max_age=settings.auth.session_max_duration_s,
        **settings.auth.status_cookie.model_dump(mode="json", exclude={"key"}),
    )

    return response


@router.post("/register")
async def basic_register(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    cache: CacheProtocol = Depends(get_cache),
):
    """Register new user"""
    # Check if registration is enabled
    if not settings.auth.basic.registration_enabled:
        raise UnauthorizedException("Registration is disabled")

    # Check email whitelist if configured
    if settings.auth.basic.email_whitelist:
        email_domain = email.split("@")[1]
        if not any(
            email.endswith(pattern) or email_domain == pattern
            for pattern in settings.auth.basic.email_whitelist
        ):
            raise UnauthorizedException("Email domain not allowed")

    from ...authorization import find_user

    existing_user = await find_user(email)
    if existing_user:
        raise UnauthorizedException("User already exists")

    from pydantic import SecretStr

    user_info = UserInfo(
        name=name, email=email, password_hash=SecretStr(hash_password(password))
    )

    # Get default teams based on email domain with role mapping
    default_team_roles = []
    email_domain = email.split("@")[1]
    if email_domain in settings.auth.basic.default_teams_by_domain:
        team_mappings = settings.auth.basic.default_teams_by_domain[email_domain]
        default_team_roles = [
            (mapping.team_name, mapping.role) for mapping in team_mappings
        ]

    session_id, session_info = await update_user_and_get_session_with_team_roles(
        user_info, cache, team_roles=default_team_roles
    )

    response = RedirectResponse(url="/", status_code=302)

    # Set main session cookie (httponly, secure)
    response.set_cookie(
        key=settings.auth.cookie.key,
        value=session_id,
        max_age=settings.auth.session_max_duration_s,
        **settings.auth.cookie.model_dump(mode="json", exclude={"key"}),
    )

    # Set session status cookie (JS readable, contains expiry timestamp)
    # Status cookie value should contain the actual session expiry (rolling), not max duration
    # Using rolling duration here for max age seems to cause the session to not renew.
    expiry_timestamp = int(time.time()) + settings.auth.session_rolling_duration_s
    response.set_cookie(
        key=settings.auth.status_cookie.key,
        value=str(expiry_timestamp),
        max_age=settings.auth.session_max_duration_s,
        **settings.auth.status_cookie.model_dump(mode="json", exclude={"key"}),
    )

    return response
