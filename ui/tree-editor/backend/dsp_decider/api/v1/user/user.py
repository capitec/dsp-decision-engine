from fastapi import Depends, HTTPException, File, UploadFile, Form
from sqlalchemy import select
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64

from dsp_decider.db.session import db_session
from dsp_decider.db.models.users import User, Team, TeamMembership
from dsp_decider.auth.models import TeamRole
from dsp_decider.auth.authorization import get_current_user_session, is_superadmin
from dsp_decider.auth.models import SessionInfo
from pydantic import BaseModel, ConfigDict
from .routes import user_router, users_router

# Profile picture constraints
MAX_PICTURE_SIZE = 2 * 1024 * 1024  # 2MB
ALLOWED_PICTURE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class UserProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: Optional[str]
    profile_picture_url: Optional[str] = None  # Base64 data URL for display
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None


class TeamMembershipResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    team_id: uuid.UUID
    team_name: str
    team_description: Optional[str]
    role: TeamRole
    joined_at: datetime


class UserTeamsResponse(BaseModel):
    teams: List[TeamMembershipResponse]


class UserSearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: Optional[str]


class UserSearchResponse(BaseModel):
    users: List[UserSearchResult]


class UserInfoResponse(BaseModel):
    is_superadmin: bool


def create_user_profile_response(user: User) -> UserProfileResponse:
    """Create a UserProfileResponse from a User model"""
    profile_picture_url = None
    if user.profile_picture:
        # Convert bytes to base64 data URL for frontend display
        base64_data = base64.b64encode(user.profile_picture).decode("utf-8")
        profile_picture_url = f"data:image/jpeg;base64,{base64_data}"

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        profile_picture_url=profile_picture_url,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@user_router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Get current user's profile information"""
    async with db_session() as session:
        query = select(User).where(User.id == current_user.user_id)
        result = await session.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return create_user_profile_response(user)


@user_router.put("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    update_data: UserUpdateRequest,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Update current user's profile information"""
    async with db_session() as session:
        query = select(User).where(User.id == current_user.user_id)
        result = await session.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update fields
        if update_data.full_name is not None:
            user.full_name = update_data.full_name

        user.updated_at = datetime.now(timezone.utc)  # Use timezone-aware datetime

        await session.commit()
        await session.refresh(user)

        return create_user_profile_response(user)


@user_router.post("/profile/picture", response_model=UserProfileResponse)
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Upload a profile picture"""
    # Validate file type
    if file.content_type not in ALLOWED_PICTURE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_PICTURE_TYPES)}",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_PICTURE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_PICTURE_SIZE // (1024*1024)}MB",
        )

    async with db_session() as session:
        query = select(User).where(User.id == current_user.user_id)
        result = await session.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update profile picture
        user.profile_picture = content
        user.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(user)

        return create_user_profile_response(user)


@user_router.delete("/profile/picture", response_model=UserProfileResponse)
async def delete_profile_picture(
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Delete profile picture"""
    async with db_session() as session:
        query = select(User).where(User.id == current_user.user_id)
        result = await session.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Remove profile picture
        user.profile_picture = None
        user.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(user)

        return create_user_profile_response(user)


@user_router.get("/teams", response_model=UserTeamsResponse)
async def get_user_teams(current_user: SessionInfo = Depends(get_current_user_session)):
    """Get current user's team memberships (or all teams for superadmins)"""
    async with db_session() as session:
        if await is_superadmin(current_user):
            # Superadmins see all teams as if they were owners
            query = select(Team)
            result = await session.execute(query)
            all_teams = result.scalars().all()

            teams = []
            for team in all_teams:
                teams.append(
                    TeamMembershipResponse(
                        id=team.id,  # Use team ID as membership ID for superadmins
                        team_id=team.id,
                        team_name=team.name,
                        team_description=team.description,
                        role=TeamRole.OWNER,  # Superadmins are shown as owners
                        joined_at=team.created_at,  # Use team creation date
                    )
                )
        else:
            # Regular users see only their teams
            query = (
                select(TeamMembership, Team.name, Team.description)
                .join(Team, TeamMembership.team_id == Team.id)
                .where(TeamMembership.user_id == current_user.user_id)
            )
            result = await session.execute(query)
            memberships_data = result.all()

            teams = []
            for membership, team_name, team_description in memberships_data:
                teams.append(
                    TeamMembershipResponse(
                        id=membership.id,
                        team_id=membership.team_id,
                        team_name=team_name,
                        team_description=team_description,
                        role=membership.role,
                        joined_at=membership.joined_at,
                    )
                )

        return UserTeamsResponse(teams=teams)


@user_router.get("/{user_id}/profile", response_model=UserProfileResponse)
async def get_other_user_profile(
    user_id: uuid.UUID, _: SessionInfo = Depends(get_current_user_session)
):
    """Get another user's public profile information (for sharing purposes)"""
    async with db_session() as session:
        query = select(User).where(User.id == user_id, User.is_active == True)
        result = await session.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Return limited profile info for privacy
        return create_user_profile_response(user)


@users_router.get("/search", response_model=UserSearchResponse)
async def search_users(
    q: str,
    limit: int = 10,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Search for active users by email or name (for team invitations)"""
    if len(q.strip()) < 2:
        raise HTTPException(
            status_code=400, detail="Search query must be at least 2 characters"
        )

    if limit > 50:
        limit = 50  # Cap the limit to prevent excessive results

    async with db_session() as session:
        # Search by email or full_name containing the query (case-insensitive)
        query = (
            select(User)
            .where(User.is_active == True)
            .where((User.email.ilike(f"%{q}%")) | (User.full_name.ilike(f"%{q}%")))
            .limit(limit)
        )
        result = await session.execute(query)
        users = result.scalars().all()

        search_results = []
        for user in users:
            search_results.append(
                UserSearchResult(id=user.id, email=user.email, full_name=user.full_name)
            )

        return UserSearchResponse(users=search_results)


@user_router.get("/info", response_model=UserInfoResponse)
async def get_user_info(current_user: SessionInfo = Depends(get_current_user_session)):
    """Get current user's information (like superadmin status)"""
    return UserInfoResponse(is_superadmin=await is_superadmin(current_user))
