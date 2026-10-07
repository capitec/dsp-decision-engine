from fastapi import Depends, HTTPException
from sqlalchemy import select
from typing import List, Optional
import uuid
from datetime import datetime

from dsp_decider.db.session import db_session
from dsp_decider.db.models.users import Team, TeamMembership, User
from dsp_decider.db.models.projects import Project, ProjectShare, AccessMode
from dsp_decider.auth.models import TeamRole
from dsp_decider.auth.authorization import get_current_user_session, is_superadmin
from dsp_decider.auth.models import SessionInfo
from dsp_decider.auth.db_service import get_user_teams, find_user_by_email
from pydantic import BaseModel, ConfigDict
from .routes import router


class TeamCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TeamUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class TeamMemberAdd(BaseModel):
    email: str
    role: TeamRole = TeamRole.MEMBER


class TeamMemberUpdate(BaseModel):
    role: TeamRole


class TeamResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str]
    created_by: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime


class TeamMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    email: str
    full_name: Optional[str]
    role: TeamRole
    joined_at: datetime


class TeamListResponse(BaseModel):
    teams: List[TeamResponse]


class TeamMembersResponse(BaseModel):
    members: List[TeamMemberResponse]


class SharedProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    access_mode: AccessMode


class TeamSharedProjectsResponse(BaseModel):
    shared_projects: List[SharedProjectResponse]


async def _check_team_view_access(
    team_id: uuid.UUID, current_user: SessionInfo
) -> Team:
    """Check if user can view team details (any team member or superadmin)"""
    async with db_session() as session:
        # Check if team exists
        result = await session.execute(select(Team).where(Team.id == team_id))
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Superadmins can view all teams
        if await is_superadmin(current_user):
            return team

        # Check if user is a member of the team (any role)
        membership_query = (
            select(TeamMembership)
            .where(TeamMembership.team_id == team_id)
            .where(TeamMembership.user_id == current_user.user_id)
        )
        membership_result = await session.execute(membership_query)
        membership = membership_result.scalar_one_or_none()

        if not membership:
            raise HTTPException(
                status_code=403, detail="Access denied - team membership required"
            )

        return team


async def _check_team_admin_access(
    team_id: uuid.UUID, current_user: SessionInfo
) -> Team:
    """Check if user has admin access to team (either superadmin or team admin/owner)"""
    async with db_session() as session:
        # Check if team exists
        result = await session.execute(select(Team).where(Team.id == team_id))
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Superadmins can manage all teams
        if await is_superadmin(current_user):
            return team

        # Check if user is team owner (only owners can manage teams)
        membership_query = (
            select(TeamMembership)
            .where(TeamMembership.team_id == team_id)
            .where(TeamMembership.user_id == current_user.user_id)
            .where(TeamMembership.role == TeamRole.OWNER)
        )
        membership_result = await session.execute(membership_query)
        membership = membership_result.scalar_one_or_none()

        if not membership:
            raise HTTPException(
                status_code=403, detail="Access denied - owner role required"
            )

        return team


@router.get("", response_model=TeamListResponse)
async def list_teams(current_user: SessionInfo = Depends(get_current_user_session)):
    """List all teams (superadmin) or user's teams (regular users)"""
    async with db_session() as session:
        if await is_superadmin(current_user):
            # Superadmins see all teams
            query = select(Team)
            result = await session.execute(query)
            teams = result.scalars().all()
        else:
            # Regular users see only their teams
            user_teams = await get_user_teams(current_user.user_id)
            query = select(Team).where(Team.name.in_(user_teams))
            result = await session.execute(query)
            teams = result.scalars().all()

        return TeamListResponse(teams=teams)


@router.post("", response_model=TeamResponse)
async def create_team(
    team_data: TeamCreate, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Create a new team (superadmin only)"""
    if not await is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmins can create teams")

    async with db_session() as session:
        # Check if team name already exists
        existing_query = select(Team).where(Team.name == team_data.name)
        existing_result = await session.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Team name already exists")

        team = Team(
            name=team_data.name,
            description=team_data.description,
            created_by=current_user.user_id,
        )

        session.add(team)
        await session.commit()
        await session.refresh(team)

        return team


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Get team details"""
    team = await _check_team_view_access(team_id, current_user)
    return team


@router.put("/{team_id}", response_model=TeamResponse)
async def update_team(
    team_id: uuid.UUID,
    team_data: TeamUpdate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Update team details (admin/owner only)"""
    team = await _check_team_admin_access(team_id, current_user)

    async with db_session() as session:
        if team_data.name is not None:
            # Check if new name already exists (excluding current team)
            existing_query = select(Team).where(
                Team.name == team_data.name, Team.id != team_id
            )
            existing_result = await session.execute(existing_query)
            if existing_result.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Team name already exists")
            team.name = team_data.name

        if team_data.description is not None:
            team.description = team_data.description

        session.add(team)
        await session.commit()
        await session.refresh(team)

        return team


@router.delete("/{team_id}")
async def delete_team(
    team_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Delete team (superadmin only)"""
    if not await is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmins can delete teams")

    async with db_session() as session:
        # Check if team exists
        result = await session.execute(select(Team).where(Team.id == team_id))
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        await session.delete(team)
        await session.commit()

        return {"message": "Team deleted successfully"}


@router.get("/{team_id}/members", response_model=TeamMembersResponse)
async def list_team_members(
    team_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """List team members (any team member can view)"""
    await _check_team_view_access(team_id, current_user)

    async with db_session() as session:
        query = (
            select(TeamMembership, User)
            .join(User, TeamMembership.user_id == User.id)
            .where(TeamMembership.team_id == team_id)
        )
        result = await session.execute(query)
        memberships = result.all()

        members = []
        for membership, user in memberships:
            members.append(
                TeamMemberResponse(
                    user_id=user.id,
                    email=user.email,
                    full_name=user.full_name,
                    role=membership.role,
                    joined_at=membership.joined_at,
                )
            )

        return TeamMembersResponse(members=members)


@router.post("/{team_id}/members", response_model=TeamMemberResponse)
async def add_team_member(
    team_id: uuid.UUID,
    member_data: TeamMemberAdd,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Add member to team (admin/owner only)"""
    await _check_team_admin_access(team_id, current_user)

    async with db_session() as session:
        # Find user by email
        user = await find_user_by_email(member_data.email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check if user is already a member
        existing_query = (
            select(TeamMembership)
            .where(TeamMembership.team_id == team_id)
            .where(TeamMembership.user_id == user.id)
        )
        existing_result = await session.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="User is already a team member")

        # Create membership
        membership = TeamMembership(
            user_id=user.id, team_id=team_id, role=member_data.role
        )

        session.add(membership)
        await session.commit()
        await session.refresh(membership)

        return TeamMemberResponse(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=membership.role,
            joined_at=membership.joined_at,
        )


@router.put("/{team_id}/members/{user_id}", response_model=TeamMemberResponse)
async def update_team_member(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    member_data: TeamMemberUpdate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Update team member role (admin/owner only)"""
    await _check_team_admin_access(team_id, current_user)

    async with db_session() as session:
        # Find membership
        membership_query = (
            select(TeamMembership, User)
            .join(User, TeamMembership.user_id == User.id)
            .where(TeamMembership.team_id == team_id)
            .where(TeamMembership.user_id == user_id)
        )
        result = await session.execute(membership_query)
        membership_user = result.first()

        if not membership_user:
            raise HTTPException(status_code=404, detail="Team member not found")

        membership, user = membership_user
        membership.role = member_data.role

        session.add(membership)
        await session.commit()
        await session.refresh(membership)

        return TeamMemberResponse(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=membership.role,
            joined_at=membership.joined_at,
        )


@router.delete("/{team_id}/members/{user_id}")
async def remove_team_member(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Remove member from team (admin/owner only)"""
    await _check_team_admin_access(team_id, current_user)

    async with db_session() as session:
        # Find membership
        membership_query = (
            select(TeamMembership)
            .where(TeamMembership.team_id == team_id)
            .where(TeamMembership.user_id == user_id)
        )
        result = await session.execute(membership_query)
        membership = result.scalar_one_or_none()

        if not membership:
            raise HTTPException(status_code=404, detail="Team member not found")

        await session.delete(membership)
        await session.commit()

        return {"message": "Team member removed successfully"}


@router.get("/{team_id}/shared-projects", response_model=TeamSharedProjectsResponse)
async def get_team_shared_projects(
    team_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Get projects shared with this team (any team member can view)"""
    await _check_team_view_access(team_id, current_user)

    async with db_session() as session:
        # Get projects shared with this team
        query = (
            select(Project, ProjectShare.access_mode)
            .join(ProjectShare, Project.id == ProjectShare.project_id)
            .where(ProjectShare.team_id == team_id)
            .order_by(Project.updated_at.desc())
        )
        result = await session.execute(query)
        projects_data = result.all()

        shared_projects = []
        for project, access_mode in projects_data:
            shared_projects.append(
                SharedProjectResponse(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    created_by=project.created_by,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                    access_mode=access_mode,
                )
            )

        return TeamSharedProjectsResponse(shared_projects=shared_projects)
