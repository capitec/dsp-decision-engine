import typing as t
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dsp_decider.db.models.users import User, Team, TeamMembership
from dsp_decider.auth.models import TeamRole
from dsp_decider.db.session import db_session, async_session_maker
from .models import UserInfo, RoleInfo
from pydantic import SecretStr

# FastAPI dependency to get database session


async def find_user_by_email(email: str) -> User | None:
    """Find a user by email address"""
    normalized_email = email.lower()
    async with db_session() as session:
        result = await session.execute(
            select(User).where(User.email == normalized_email)
        )
        return result.scalar_one_or_none()


async def find_user_by_id(user_id: UUID) -> User | None:
    """Find a user by ID"""
    async with db_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


async def create_user(user_info: UserInfo) -> User:
    """Create a new user"""
    async with db_session() as session:
        user = User(
            id=user_info.user_id,
            email=user_info.email.lower(),  # Always store email as lowercase
            full_name=user_info.name,
            password_hash=(
                user_info.password_hash.get_secret_value()
                if user_info.password_hash
                else None
            ),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def update_user(user_id: UUID, user_info: UserInfo) -> User:
    """Update an existing user"""
    async with db_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        user.full_name = user_info.name
        user.email = user_info.email.lower()  # Always store email as lowercase

        await session.commit()
        await session.refresh(user)
        return user


async def get_user_teams(user_id: UUID) -> t.List[str]:
    """Get all team names for a user"""
    async with db_session() as session:
        # Get user with team memberships
        result = await session.execute(
            select(User)
            .options(
                selectinload(User.team_memberships).selectinload(TeamMembership.team)
            )
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            return []

        teams = []
        for membership in user.team_memberships:
            teams.append(membership.team.name)

        return teams


async def is_user_superadmin(user_id: UUID, superadmin_teams: t.List[str]) -> bool:
    """Check if user is superadmin based on team membership"""
    user_teams = await get_user_teams(user_id)
    return any(team in superadmin_teams for team in user_teams)


async def find_team_by_name(name: str) -> Team | None:
    """Find a team by name"""
    async with db_session() as session:
        result = await session.execute(select(Team).where(Team.name == name))
        return result.scalar_one_or_none()


async def create_team(
    name: str, description: str | None = None, created_by: UUID | None = None
) -> Team:
    """Create a new team"""
    async with db_session() as session:
        team = Team(
            name=name,
            description=description,
            created_by=created_by,
        )
        session.add(team)
        await session.commit()
        await session.refresh(team)
        return team


async def add_user_to_team(
    user_id: UUID, team_name: str, role: TeamRole = TeamRole.MEMBER
) -> TeamMembership:
    """Add a user to a team"""
    async with db_session() as session:
        # Find the team
        team = await find_team_by_name(team_name)
        if not team:
            # Create the team if it doesn't exist
            team = await create_team(team_name)

        # Check if membership already exists
        result = await session.execute(
            select(TeamMembership)
            .where(TeamMembership.user_id == user_id)
            .where(TeamMembership.team_id == team.id)
        )
        existing_membership = result.scalar_one_or_none()

        if existing_membership:
            # Update role if different
            if existing_membership.role != role:
                existing_membership.role = role
                await session.commit()
                await session.refresh(existing_membership)
            return existing_membership

        # Create new membership
        membership = TeamMembership(user_id=user_id, team_id=team.id, role=role)
        session.add(membership)
        await session.commit()
        await session.refresh(membership)
        return membership


async def upsert_user_with_teams(
    user_info: UserInfo, teams: t.List[str] | None = None
) -> User:
    """Create or update a user and handle their team assignments"""
    # Check if user exists
    existing_user = await find_user_by_email(user_info.email)

    if existing_user:
        # Update existing user
        user = await update_user(existing_user.id, user_info)
    else:
        # Create new user
        user = await create_user(user_info)

    # Handle team assignments if provided
    if teams is not None:
        for team_name in teams:
            await add_user_to_team(user.id, team_name)

    return user


async def upsert_user_with_team_roles(
    user_info: UserInfo, team_roles: t.List[t.Tuple[str, TeamRole]] | None = None
) -> User:
    """Create or update a user and handle their team assignments with specific roles"""
    # Check if user exists
    existing_user = await find_user_by_email(user_info.email)

    if existing_user:
        # Update existing user
        user = await update_user(existing_user.id, user_info)
    else:
        # Create new user
        user = await create_user(user_info)

    # Handle team assignments with roles if provided
    if team_roles is not None:
        for team_name, role in team_roles:
            await add_user_to_team(user.id, team_name, role)

    return user


# Legacy function for backward compatibility
async def upsert_user_with_roles(
    user_info: UserInfo, roles: t.List[str] | None = None
) -> User:
    """Legacy function - now maps to team-based approach"""
    # Convert roles to teams (roles are now team names)
    return await upsert_user_with_teams(user_info, roles)


# Legacy compatibility functions for gradual migration
def user_to_userinfo(user: User) -> UserInfo:
    """Convert database User to UserInfo model"""
    return UserInfo(
        user_id=user.id,
        name=user.full_name or "",
        email=user.email,
        created=user.created_at,
        password_hash=SecretStr(user.password_hash) if user.password_hash else None,
    )


def role_to_roleinfo(role_name: str) -> RoleInfo:
    """Convert role name to RoleInfo model"""
    return RoleInfo(name=role_name)
