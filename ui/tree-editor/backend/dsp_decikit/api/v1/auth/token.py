import typing as t
import uuid
from datetime import datetime, timezone
from fastapi import Depends, HTTPException
from sqlalchemy import select, delete
from pydantic import BaseModel, Field

from dsp_decikit.db.session import db_session
from dsp_decikit.db.models.users import ApiToken
from dsp_decikit.auth.models import TokenScope
from dsp_decikit.auth.authorization import get_current_user_session, generate_api_token
from dsp_decikit.auth.models import SessionInfo
from .routes import router


# Request/Response Models
class TokenCreate(BaseModel):
    name: str = Field(
        min_length=1, max_length=255, description="Human-readable name for the token"
    )
    scopes: t.List[TokenScope] = Field(
        default=[TokenScope.READ], description="Token permissions"
    )
    expires_at: t.Optional[datetime] = Field(
        None, description="Optional expiration date"
    )


class TokenResponse(BaseModel):
    id: uuid.UUID
    name: str
    token_prefix: str
    scopes: t.List[TokenScope]
    is_active: bool
    last_used_at: t.Optional[datetime]
    expires_at: t.Optional[datetime]
    created_at: datetime


class TokenCreateResponse(BaseModel):
    token: str  # Full token - only returned once!
    token_info: TokenResponse


class TokenUpdate(BaseModel):
    name: t.Optional[str] = Field(None, min_length=1, max_length=255)
    is_active: t.Optional[bool] = None
    expires_at: t.Optional[datetime] = None


class TokenListResponse(BaseModel):
    tokens: t.List[TokenResponse]


# API Endpoints
@router.post("", response_model=TokenCreateResponse)
async def create_api_token(
    token_data: TokenCreate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Create a new API token for the authenticated user"""

    # Security: Always create token for the authenticated user only
    raw_token, token_prefix, token_hash = generate_api_token()

    async with db_session() as session:
        # Create new token with timezone-aware datetimes
        api_token = ApiToken(
            user_id=current_user.user_id,
            name=token_data.name,
            token_hash=token_hash,
            token_prefix=token_prefix,
            scopes=token_data.scopes,
            expires_at=token_data.expires_at,  # Keep timezone-aware datetime from frontend
            is_active=True,
            created_at=datetime.now(timezone.utc),  # Use timezone-aware UTC datetime
        )

        session.add(api_token)
        await session.commit()
        await session.refresh(api_token)

        # Return response with full token (only time it's shown!)
        return TokenCreateResponse(
            token=raw_token,
            token_info=TokenResponse(
                id=api_token.id,
                name=api_token.name,
                token_prefix=api_token.token_prefix,
                scopes=api_token.scopes,
                is_active=api_token.is_active,
                last_used_at=api_token.last_used_at,
                expires_at=api_token.expires_at,
                created_at=api_token.created_at,
            ),
        )


@router.get("", response_model=TokenListResponse)
async def list_api_tokens(
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """List all API tokens for the authenticated user"""

    async with db_session() as session:
        query = select(ApiToken).where(ApiToken.user_id == current_user.user_id)
        result = await session.execute(query)
        tokens = result.scalars().all()

        return TokenListResponse(
            tokens=[
                TokenResponse(
                    id=token.id,
                    name=token.name,
                    token_prefix=token.token_prefix,
                    scopes=token.scopes,
                    is_active=token.is_active,
                    last_used_at=token.last_used_at,
                    expires_at=token.expires_at,
                    created_at=token.created_at,
                )
                for token in tokens
            ]
        )


@router.get("/{token_id}", response_model=TokenResponse)
async def get_api_token(
    token_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Get details of a specific API token"""

    async with db_session() as session:
        query = select(ApiToken).where(
            ApiToken.id == token_id, ApiToken.user_id == current_user.user_id
        )
        result = await session.execute(query)
        token = result.scalar_one_or_none()

        if not token:
            raise HTTPException(status_code=404, detail="Token not found")

        return TokenResponse(
            id=token.id,
            name=token.name,
            token_prefix=token.token_prefix,
            scopes=token.scopes,
            is_active=token.is_active,
            last_used_at=token.last_used_at,
            expires_at=token.expires_at,
            created_at=token.created_at,
        )


@router.put("/{token_id}", response_model=TokenResponse)
async def update_api_token(
    token_id: uuid.UUID,
    token_update: TokenUpdate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Update an API token (name, active status, expiration)"""

    async with db_session() as session:
        # Verify token belongs to user
        query = select(ApiToken).where(
            ApiToken.id == token_id, ApiToken.user_id == current_user.user_id
        )
        result = await session.execute(query)
        token = result.scalar_one_or_none()

        if not token:
            raise HTTPException(status_code=404, detail="Token not found")

        # Update fields
        if token_update.name is not None:
            token.name = token_update.name
        if token_update.is_active is not None:
            token.is_active = token_update.is_active
        if token_update.expires_at is not None:
            token.expires_at = token_update.expires_at

        await session.commit()
        await session.refresh(token)

        return TokenResponse(
            id=token.id,
            name=token.name,
            token_prefix=token.token_prefix,
            scopes=token.scopes,
            is_active=token.is_active,
            last_used_at=token.last_used_at,
            expires_at=token.expires_at,
            created_at=token.created_at,
        )


@router.delete("/{token_id}")
async def delete_api_token(
    token_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Delete/revoke an API token"""

    async with db_session() as session:
        # Verify token belongs to user and delete
        query = delete(ApiToken).where(
            ApiToken.id == token_id, ApiToken.user_id == current_user.user_id
        )
        result = await session.execute(query)

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Token not found")

        await session.commit()

        return {"message": "Token deleted successfully"}
