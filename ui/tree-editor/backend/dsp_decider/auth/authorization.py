import typing as t
import time
import bcrypt
import secrets
from datetime import datetime, timezone
from fastapi import Depends, status as http_status
from fastapi.exceptions import HTTPException
from fastapi.security import APIKeyCookie, HTTPBearer
from dsp_decider.settings import settings
from dsp_decider.cache import get_cache, CacheProtocol, CacheKeys
from dsp_decider.db.session import db_session
import logging
from .models import SessionInfo, UserInfo, BearerTokenSessionInfo
from .db_service import (
    find_user_by_email,
    user_to_userinfo,
    upsert_user_with_roles,
    upsert_user_with_team_roles,
    get_user_teams as get_user_teams_from_db,
    is_user_superadmin,
)
from dsp_decider.db.models.users import ApiToken
from dsp_decider.auth.models import TeamRole
from sqlalchemy import select
from functools import lru_cache

logger = logging.getLogger(__name__)

auth_settings = settings.auth

session_token_auth = APIKeyCookie(name=auth_settings.cookie.key, auto_error=False)
bearer_token_auth = HTTPBearer(auto_error=False)


# Just make life a bit quicker for common users
@lru_cache(maxsize=512)
def hash_api_token(raw_token: str) -> bytes:
    return bcrypt.kdf(
        password=raw_token.encode(),
        salt=auth_settings.api_key.salt_value.get_secret_value(),
        desired_key_bytes=auth_settings.api_key.TOKEN_HASH_LENGTH,
        rounds=auth_settings.api_key.rounds,
    )


def generate_api_token() -> t.Tuple[str, str, bytes]:
    """Generate API token, prefix, and hash"""
    # Generate secure token
    raw_token = secrets.token_urlsafe(auth_settings.api_key.token_length)

    # Create prefix for display
    token_prefix = raw_token[: auth_settings.api_key.PREVIEW_LENGTH]  # "dsp_a1b2c3d4"

    # Hash for storage
    token_hash = hash_api_token(raw_token)

    return raw_token, token_prefix, token_hash


async def handle_session_auth(session_token: str, cache: CacheProtocol):
    if not session_token:
        return None
    cache_key = CacheKeys.auth_session(session_token)
    session_info = await cache.get(cache_key, None)
    if session_info is None:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Invalid Session"
        )
    try:
        session_info_model = SessionInfo.model_validate(session_info)
    except ValueError as e:
        logger.error(e)
        await cache.delete(cache_key)
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Unknown issue with Session",
        )
    return session_info_model


async def handle_bearer_token_auth(bearer_token: str) -> SessionInfo | None:
    """Handle Bearer token authentication using API tokens"""
    if not bearer_token:
        return None

    # Hash the provided token to compare with stored hash
    try:
        token_hash = hash_api_token(bearer_token)
    except Exception as e:
        logger.error(f"Failed to hash bearer token: {e}")
        return None

    async with db_session() as session:
        try:
            # Look up token by hash
            query = select(ApiToken).where(
                ApiToken.token_hash == token_hash, ApiToken.is_active == True
            )
            result = await session.execute(query)
            api_token = result.scalar_one_or_none()

            if not api_token:
                return None

            # Check expiration (compare timezone-aware datetimes)
            current_time = datetime.now(timezone.utc)
            if api_token.expires_at and api_token.expires_at <= current_time:
                return None

            # Update last used timestamp with timezone-aware datetime
            api_token.last_used_at = current_time
            await session.commit()

            # Return bearer token session info with API token's expiry
            current_timestamp = int(time.time())
            token_expires_at = (
                int(api_token.expires_at.timestamp())
                if api_token.expires_at
                else int(time.time()) + 86400 * 365  # Default 1 year if no expiration
            )
            return BearerTokenSessionInfo(
                created_at=current_timestamp,
                last_activity=current_timestamp,
                user_id=api_token.user_id,
                expires_at=token_expires_at,
            )

        except Exception as e:
            logger.error(f"Database error in bearer token auth: {e}")
            return None


async def find_user(email: str) -> UserInfo | None:
    """Find a user by email - now using database"""
    user = await find_user_by_email(email)
    return user_to_userinfo(user) if user else None


async def _upsert_user(
    user_info: UserInfo, roles: t.List[str] | None = None
) -> UserInfo:
    """Create or update user - now using database"""
    user = await upsert_user_with_roles(user_info, roles)
    return user_to_userinfo(user)


async def _upsert_user_with_team_roles(
    user_info: UserInfo, team_roles: t.List[t.Tuple[str, TeamRole]] | None = None
) -> UserInfo:
    """Create or update user with team roles - now using database"""
    user = await upsert_user_with_team_roles(user_info, team_roles)
    return user_to_userinfo(user)


async def update_user_and_get_session(
    user_info: UserInfo,
    cache: CacheProtocol,
    roles: t.List[str] | None = None,
):
    user = await _upsert_user(user_info, roles)

    # Create new session
    session_id = secrets.token_urlsafe(auth_settings.session_token_len)

    # Use timezone-aware datetime for session creation
    current_time = datetime.now(timezone.utc)
    current_timestamp = int(current_time.timestamp())

    session_info = SessionInfo(
        created_at=current_timestamp,
        last_activity=current_timestamp,
        user_id=user.user_id,
    )

    await cache.put(
        CacheKeys.auth_session(session_id),
        session_info.model_dump(mode="json"),
        expire_seconds=auth_settings.session_max_duration_s,  # Use max duration for cache TTL
    )

    return session_id, session_info


async def update_user_and_get_session_with_team_roles(
    user_info: UserInfo,
    cache: CacheProtocol,
    team_roles: t.List[t.Tuple[str, TeamRole]] | None = None,
):
    user = await _upsert_user_with_team_roles(user_info, team_roles)

    # Create new session
    session_id = secrets.token_urlsafe(auth_settings.session_token_len)

    # Use timezone-aware datetime for session creation
    current_time = datetime.now(timezone.utc)
    current_timestamp = int(current_time.timestamp())

    session_info = SessionInfo(
        created_at=current_timestamp,
        last_activity=current_timestamp,
        user_id=user.user_id,
    )

    await cache.put(
        CacheKeys.auth_session(session_id),
        session_info.model_dump(mode="json"),
        expire_seconds=auth_settings.session_max_duration_s,  # Use max duration for cache TTL
    )

    return session_id, session_info


# This is the function that will be used to authenticate all requests and get the session info
async def security(
    session_token: str | None = Depends(session_token_auth),
    bearer_token: str | None = Depends(bearer_token_auth),
    cache: CacheProtocol = Depends(get_cache),
):
    # Authentication priority: session tokens (web UI) -> API tokens (Bearer) -> legacy API key
    session_info = (await handle_session_auth(session_token, cache)) or (
        await handle_bearer_token_auth(
            bearer_token.credentials if bearer_token else None
        )
    )

    if not session_info:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Not Authorized"
        )

    # Check session expiration using current UTC time
    current_timestamp = datetime.now(timezone.utc).timestamp()
    if session_info.expires_at <= current_timestamp:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Session Expired"
        )

    return session_info


async def has_team_access(
    session_info: SessionInfo, required_teams: t.List[str]
) -> bool:
    """Check if user has access to any of the required teams"""
    if await is_user_superadmin(session_info.user_id, auth_settings.superadmin_teams):
        return True
    user_teams = await get_user_teams_from_db(session_info.user_id)
    return any(team in user_teams for team in required_teams)


async def is_superadmin(session_info: SessionInfo) -> bool:
    """Check if user is superadmin"""
    return await is_user_superadmin(
        session_info.user_id, auth_settings.superadmin_teams
    )


async def get_user_teams(session_info: SessionInfo) -> t.List[str]:
    """Get list of teams the user belongs to"""
    return await get_user_teams_from_db(session_info.user_id)


# Convenience alias for FastAPI dependencies
get_current_user_session = security
