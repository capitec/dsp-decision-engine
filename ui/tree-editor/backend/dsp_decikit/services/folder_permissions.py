"""Folder permission resolution service.

Implements hierarchical permission inheritance:
- Projects inherit from folders
- Folders inherit from parent folders
- Always takes the HIGHEST permission (READ < WRITE < ADMIN)
"""

import typing as t
import uuid
from enum import Enum

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from dsp_decikit.db.models.projects import AccessMode, ShareType, Project, ProjectShare
from dsp_decikit.db.models.folders import Folder, FolderShare
from dsp_decikit.db.models.users import TeamMembership
from dsp_decikit.auth.models import SessionInfo
from dsp_decikit.auth.authorization import is_superadmin


class PermissionSource(str, Enum):
    """Source of a permission."""

    OWNERSHIP = "ownership"
    SUPERADMIN = "superadmin"
    DIRECT_SHARE = "direct_share"
    FOLDER_SHARE = "folder_share"
    TEAM_SHARE = "team_share"


class PermissionInfo(t.NamedTuple):
    """Details about a specific permission grant."""

    source: PermissionSource
    access_mode: AccessMode
    source_id: t.Optional[uuid.UUID] = None  # Share ID, folder ID, or team ID
    source_name: t.Optional[str] = None  # Folder name or team name
    folder_path: t.Optional[str] = None  # Path for inherited folder permissions


async def get_user_teams(
    session: AsyncSession, user_id: uuid.UUID
) -> t.List[uuid.UUID]:
    """Get list of team IDs the user belongs to."""
    stmt = select(TeamMembership.team_id).where(TeamMembership.user_id == user_id)
    result = await session.execute(stmt)
    return [row[0] for row in result.fetchall()]


async def get_folder_hierarchy(
    session: AsyncSession, folder_id: t.Optional[uuid.UUID]
) -> t.List[Folder]:
    """Get folder hierarchy from current folder to root.

    Returns list of folders from current to root (ascending order).
    """
    if folder_id is None:
        return []

    folders = []
    current_id = folder_id

    # Traverse up the tree (limit to 100 levels to prevent infinite loops)
    for _ in range(100):
        if current_id is None:
            break

        stmt = select(Folder).where(Folder.id == current_id)
        result = await session.execute(stmt)
        folder = result.scalar_one_or_none()

        if folder is None:
            break

        folders.append(folder)
        current_id = folder.parent_id

    return folders


async def get_project_shares(
    session: AsyncSession,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    team_ids: t.List[uuid.UUID],
) -> t.List[t.Tuple[ProjectShare, PermissionSource]]:
    """Get all project shares for user (direct and via teams)."""
    shares_with_source = []

    # Direct user shares
    stmt = select(ProjectShare).where(
        ProjectShare.project_id == project_id,
        ProjectShare.share_type == ShareType.USER,
        ProjectShare.user_id == user_id,
    )
    result = await session.execute(stmt)
    for share in result.scalars():
        shares_with_source.append((share, PermissionSource.DIRECT_SHARE))

    # Team shares
    if team_ids:
        stmt = select(ProjectShare).where(
            ProjectShare.project_id == project_id,
            ProjectShare.share_type == ShareType.TEAM,
            ProjectShare.team_id.in_(team_ids),
        )
        result = await session.execute(stmt)
        for share in result.scalars():
            shares_with_source.append((share, PermissionSource.TEAM_SHARE))

    return shares_with_source


async def get_folder_shares(
    session: AsyncSession,
    folder_id: uuid.UUID,
    user_id: uuid.UUID,
    team_ids: t.List[uuid.UUID],
) -> t.List[t.Tuple[FolderShare, PermissionSource]]:
    """Get all folder shares for user (direct and via teams)."""
    shares_with_source = []

    # Direct user shares
    stmt = select(FolderShare).where(
        FolderShare.folder_id == folder_id,
        FolderShare.share_type == ShareType.USER,
        FolderShare.user_id == user_id,
    )
    result = await session.execute(stmt)
    for share in result.scalars():
        shares_with_source.append((share, PermissionSource.FOLDER_SHARE))

    # Team shares
    if team_ids:
        stmt = select(FolderShare).where(
            FolderShare.folder_id == folder_id,
            FolderShare.share_type == ShareType.TEAM,
            FolderShare.team_id.in_(team_ids),
        )
        result = await session.execute(stmt)
        for share in result.scalars():
            shares_with_source.append((share, PermissionSource.FOLDER_SHARE))

    return shares_with_source


def resolve_access_mode(modes: t.List[AccessMode]) -> AccessMode:
    """Return the highest permission from a list of access modes.

    Hierarchy: READ < WRITE < ADMIN
    """
    if not modes:
        return AccessMode.READ

    if AccessMode.ADMIN in modes:
        return AccessMode.ADMIN
    if AccessMode.WRITE in modes:
        return AccessMode.WRITE
    return AccessMode.READ


async def get_effective_project_permissions(
    session: AsyncSession,
    project_id: uuid.UUID,
    current_user: SessionInfo,
) -> t.Tuple[bool, AccessMode, t.List[PermissionInfo]]:
    """Get effective permissions for a project.

    Returns:
        (is_owner, effective_access_mode, permission_details)
    """
    user_id = current_user.user_id

    # Get project
    stmt = select(Project).where(Project.id == project_id)
    result = await session.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise ValueError(f"Project {project_id} not found")

    permissions: t.List[PermissionInfo] = []

    # 1. Check ownership
    is_owner = project.created_by == user_id
    if is_owner:
        permissions.append(
            PermissionInfo(
                source=PermissionSource.OWNERSHIP,
                access_mode=AccessMode.ADMIN,
            )
        )
        return True, AccessMode.ADMIN, permissions

    # 2. Check superadmin
    if await is_superadmin(current_user):
        permissions.append(
            PermissionInfo(
                source=PermissionSource.SUPERADMIN,
                access_mode=AccessMode.ADMIN,
            )
        )
        return False, AccessMode.ADMIN, permissions

    # 3. Get user's teams
    team_ids = await get_user_teams(session, user_id)

    # 4. Get direct project shares
    project_shares = await get_project_shares(session, project_id, user_id, team_ids)
    for share, source in project_shares:
        permissions.append(
            PermissionInfo(
                source=source,
                access_mode=share.access_mode,
                source_id=share.id,
            )
        )

    # 5. If project has folder, get folder hierarchy permissions
    if project.folder_id:
        folders = await get_folder_hierarchy(session, project.folder_id)

        for folder in folders:
            folder_shares = await get_folder_shares(
                session, folder.id, user_id, team_ids
            )

            for share, _ in folder_shares:
                permissions.append(
                    PermissionInfo(
                        source=PermissionSource.FOLDER_SHARE,
                        access_mode=share.access_mode,
                        source_id=folder.id,
                        source_name=folder.name,
                    )
                )

    # 6. Resolve to highest permission
    if not permissions:
        raise PermissionError("User has no access to this project")

    access_modes = [p.access_mode for p in permissions]
    effective_mode = resolve_access_mode(access_modes)

    return False, effective_mode, permissions


async def get_effective_folder_permissions(
    session: AsyncSession,
    folder_id: uuid.UUID,
    current_user: SessionInfo,
) -> t.Tuple[bool, AccessMode, t.List[PermissionInfo]]:
    """Get effective permissions for a folder.

    Returns:
        (is_owner, effective_access_mode, permission_details)
    """
    user_id = current_user.user_id

    # Get folder
    stmt = select(Folder).where(Folder.id == folder_id)
    result = await session.execute(stmt)
    folder = result.scalar_one_or_none()

    if folder is None:
        raise ValueError(f"Folder {folder_id} not found")

    permissions: t.List[PermissionInfo] = []

    # 1. Check ownership
    is_owner = folder.created_by == user_id
    if is_owner:
        permissions.append(
            PermissionInfo(
                source=PermissionSource.OWNERSHIP,
                access_mode=AccessMode.ADMIN,
            )
        )
        return True, AccessMode.ADMIN, permissions

    # 2. Check superadmin
    if await is_superadmin(current_user):
        permissions.append(
            PermissionInfo(
                source=PermissionSource.SUPERADMIN,
                access_mode=AccessMode.ADMIN,
            )
        )
        return False, AccessMode.ADMIN, permissions

    # 3. Get user's teams
    team_ids = await get_user_teams(session, user_id)

    # 4. Get direct folder shares
    folder_shares = await get_folder_shares(session, folder_id, user_id, team_ids)
    for share, source in folder_shares:
        permissions.append(
            PermissionInfo(
                source=source,
                access_mode=share.access_mode,
                source_id=share.id,
            )
        )

    # 5. Get parent folder permissions (traverse hierarchy)
    if folder.parent_id:
        parent_folders = await get_folder_hierarchy(session, folder.parent_id)

        for parent in parent_folders:
            parent_shares = await get_folder_shares(
                session, parent.id, user_id, team_ids
            )

            for share, _ in parent_shares:
                permissions.append(
                    PermissionInfo(
                        source=PermissionSource.FOLDER_SHARE,
                        access_mode=share.access_mode,
                        source_id=parent.id,
                        source_name=parent.name,
                    )
                )

    # 6. Resolve to highest permission
    if not permissions:
        raise PermissionError("User has no access to this folder")

    access_modes = [p.access_mode for p in permissions]
    effective_mode = resolve_access_mode(access_modes)

    return False, effective_mode, permissions


async def check_project_access(
    session: AsyncSession,
    project_id: uuid.UUID,
    current_user: SessionInfo,
    required_mode: AccessMode = AccessMode.READ,
) -> t.Tuple[Project, bool, AccessMode]:
    """Check if user has required access to project.

    Returns:
        (project, is_owner, effective_access_mode)

    Raises:
        PermissionError: If user doesn't have required access
    """
    is_owner, access_mode, _ = await get_effective_project_permissions(
        session, project_id, current_user
    )

    # Check if user has sufficient permissions
    mode_hierarchy = {
        AccessMode.READ: 1,
        AccessMode.WRITE: 2,
        AccessMode.ADMIN: 3,
    }

    if mode_hierarchy[access_mode] < mode_hierarchy[required_mode]:
        raise PermissionError(
            f"User requires {required_mode.value} access but only has {access_mode.value}"
        )

    # Get project
    stmt = select(Project).where(Project.id == project_id)
    result = await session.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise ValueError(f"Project {project_id} not found")

    return project, is_owner, access_mode


async def check_folder_access(
    session: AsyncSession,
    folder_id: uuid.UUID,
    current_user: SessionInfo,
    required_mode: AccessMode = AccessMode.READ,
) -> t.Tuple[Folder, bool, AccessMode]:
    """Check if user has required access to folder.

    Returns:
        (folder, is_owner, effective_access_mode)

    Raises:
        PermissionError: If user doesn't have required access
    """
    is_owner, access_mode, _ = await get_effective_folder_permissions(
        session, folder_id, current_user
    )

    # Check if user has sufficient permissions
    mode_hierarchy = {
        AccessMode.READ: 1,
        AccessMode.WRITE: 2,
        AccessMode.ADMIN: 3,
    }

    if mode_hierarchy[access_mode] < mode_hierarchy[required_mode]:
        raise PermissionError(
            f"User requires {required_mode.value} access but only has {access_mode.value}"
        )

    # Get folder
    stmt = select(Folder).where(Folder.id == folder_id)
    result = await session.execute(stmt)
    folder = result.scalar_one_or_none()

    if folder is None:
        raise ValueError(f"Folder {folder_id} not found")

    return folder, is_owner, access_mode
