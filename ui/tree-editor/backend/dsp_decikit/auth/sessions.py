"""Sign-in sessions and OIDC login state, kept in Postgres so every backend node sees the same ones at once."""

import hashlib
import secrets
import time
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select, update

from dsp_decikit.cache.keys import CacheKeys
from dsp_decikit.db.models.cache import CacheEntry
from dsp_decikit.db.session import db_session
from dsp_decikit.settings import settings

from .models import SessionInfo


def _hashed(token: str) -> str:
    # Only a hash is stored, so reading the table does not hand out usable sessions.
    return hashlib.sha256(token.encode()).hexdigest()


def _at(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, timezone.utc)


async def _insert(key: str, value: dict, expires_at: datetime) -> None:
    async with db_session() as session:
        session.add(CacheEntry(key=key, value=value, expires_at=expires_at))
        await session.commit()
        # SKIP LOCKED: two nodes clearing at once neither wait for nor deadlock on each other.
        stale = select(CacheEntry.key).where(CacheEntry.expires_at < _at(time.time())).with_for_update(skip_locked=True)
        await session.execute(delete(CacheEntry).where(CacheEntry.key.in_(stale)))
        await session.commit()


async def create_session(user_id: UUID) -> tuple[str, SessionInfo]:
    token = secrets.token_urlsafe(settings.auth.session_token_len)
    now = int(time.time())
    info = SessionInfo(created_at=now, last_activity=now, user_id=user_id)
    await _insert(CacheKeys.auth_session(_hashed(token)), info.model_dump(mode="json"), _at(info.expires_at))
    return token, info


async def get_session(token: str) -> SessionInfo | None:
    async with db_session() as session:
        value = await session.scalar(
            select(CacheEntry.value).where(CacheEntry.key == CacheKeys.auth_session(_hashed(token)))
        )
    return None if value is None else SessionInfo.model_validate(value)


async def renew_session(token: str, info: SessionInfo) -> bool:
    # UPDATE, never upsert: a renewal racing a sign-out on another node must not bring the session back.
    async with db_session() as session:
        result = await session.execute(
            update(CacheEntry)
            .where(CacheEntry.key == CacheKeys.auth_session(_hashed(token)))
            .values(value=info.model_dump(mode="json"), expires_at=_at(info.expires_at))
        )
        await session.commit()
    return result.rowcount == 1


async def delete_session(token: str) -> None:
    async with db_session() as session:
        await session.execute(delete(CacheEntry).where(CacheEntry.key == CacheKeys.auth_session(_hashed(token))))
        await session.commit()


async def put_login_state(login_id: str, state: dict, timeout_s: int) -> None:
    await _insert(CacheKeys.oidc_login_session(_hashed(login_id)), state, _at(time.time() + timeout_s))


async def take_login_state(login_id: str) -> dict | None:
    """The state of an unexpired OIDC login, removed as it is read so it works once, on whichever node."""
    async with db_session() as session:
        value = await session.scalar(
            delete(CacheEntry)
            .where(CacheEntry.key == CacheKeys.oidc_login_session(_hashed(login_id)),
                   CacheEntry.expires_at > _at(time.time()))
            .returning(CacheEntry.value)
        )
        await session.commit()
    return value
