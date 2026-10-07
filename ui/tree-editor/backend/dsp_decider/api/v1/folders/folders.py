"""Folder management API endpoints."""

import typing as t
import uuid
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Query, APIRouter
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import selectinload

from dsp_decider.db.session import db_session
from dsp_decider.db.models.folders import Folder, FolderShare
from dsp_decider.db.models.projects import Project, AccessMode, ShareType
from dsp_decider.db.models.users import User, Team
from dsp_decider.auth.authorization import get_current_user_session
from dsp_decider.auth.models import SessionInfo
from dsp_decider.services.folder_permissions import (
    check_folder_access,
    get_effective_folder_permissions,
    get_folder_hierarchy,
    get_user_teams,  # Returns team IDs (UUIDs), not names
)

from .routes import router
from .schemas import (
    FolderCreate,
    FolderUpdate,
    FolderMove,
    FolderShareCreate,
    FolderResponse,
    FolderDetailResponse,
    FolderShareResponse,
    FolderBreadcrumb,
    EffectivePermissions,
    PermissionSourceDetail,
)


async def _build_breadcrumb(
    session, folder_id: t.Optional[uuid.UUID]
) -> t.List[FolderBreadcrumb]:
    """Build breadcrumb path from root to folder."""
    if folder_id is None:
        return []

    folders = await get_folder_hierarchy(session, folder_id)
    # Reverse to go from root to current
    return [FolderBreadcrumb(id=f.id, name=f.name) for f in reversed(folders)]


async def _check_circular_reference(
    session, folder_id: uuid.UUID, new_parent_id: t.Optional[uuid.UUID]
) -> bool:
    """Check if moving folder would create a circular reference.

    Returns True if circular reference detected.
    """
    if new_parent_id is None:
        return False

    if folder_id == new_parent_id:
        return True

    # Traverse up from new_parent_id and check if we encounter folder_id
    hierarchy = await get_folder_hierarchy(session, new_parent_id)
    return any(f.id == folder_id for f in hierarchy)


@router.get("", response_model=t.List[FolderResponse])
async def list_folders(
    parent_id: t.Optional[uuid.UUID] = Query(
        None, description="Filter by parent folder (null for root)"
    ),
    search: t.Optional[str] = Query(None, description="Search by folder name"),
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """List all accessible folders.

    Returns folders the user owns or has shared access to.
    Does NOT paginate - returns all folders.
    """
    async with db_session() as session:
        user_id = current_user.user_id
        team_ids = await get_user_teams(session, user_id)

        # Build query for accessible folders
        stmt = select(Folder).distinct()

        # Filter by parent
        if parent_id is not None:
            stmt = stmt.where(Folder.parent_id == parent_id)
        else:
            # If parent_id query param not provided, return all accessible folders
            pass

        # Search filter
        if search:
            stmt = stmt.where(Folder.name.ilike(f"%{search}%"))

        # Access filter: owned OR shared
        stmt = stmt.where(
            or_(
                Folder.created_by == user_id,
                Folder.id.in_(
                    select(FolderShare.folder_id).where(
                        or_(
                            and_(
                                FolderShare.share_type == ShareType.USER,
                                FolderShare.user_id == user_id,
                            ),
                            (
                                and_(
                                    FolderShare.share_type == ShareType.TEAM,
                                    FolderShare.team_id.in_(team_ids),
                                )
                                if team_ids
                                else False
                            ),
                        )
                    )
                ),
            )
        )

        # Order by name
        stmt = stmt.order_by(Folder.name)

        result = await session.execute(stmt)
        folders = result.scalars().all()

        # Build response with computed fields
        response = []
        for folder in folders:
            # Count projects in this folder
            project_count_stmt = select(func.count(Project.id)).where(
                Project.folder_id == folder.id
            )
            project_count_result = await session.execute(project_count_stmt)
            project_count = project_count_result.scalar() or 0

            # Count subfolders
            subfolder_count_stmt = select(func.count(Folder.id)).where(
                Folder.parent_id == folder.id
            )
            subfolder_count_result = await session.execute(subfolder_count_stmt)
            subfolder_count = subfolder_count_result.scalar() or 0

            # Check if shared
            share_count_stmt = select(func.count(FolderShare.id)).where(
                FolderShare.folder_id == folder.id
            )
            share_count_result = await session.execute(share_count_stmt)
            is_shared = (share_count_result.scalar() or 0) > 0

            # Get effective permissions
            try:
                is_owner, access_mode, _ = await get_effective_folder_permissions(
                    session, folder.id, current_user
                )
            except (PermissionError, ValueError):
                # Skip folders user doesn't have access to
                continue

            response.append(
                FolderResponse(
                    id=folder.id,
                    name=folder.name,
                    description=folder.description,
                    parent_id=folder.parent_id,
                    created_by=folder.created_by,
                    created_at=folder.created_at,
                    updated_at=folder.updated_at,
                    project_count=project_count,
                    subfolder_count=subfolder_count,
                    is_owner=is_owner,
                    access_mode=access_mode,
                    is_shared=is_shared,
                )
            )

        return response


@router.post("", response_model=FolderResponse, status_code=201)
async def create_folder(
    folder_data: FolderCreate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Create a new folder.

    If parent_id is provided, user must have WRITE or ADMIN access to parent.
    """
    async with db_session() as session:
        user_id = current_user.user_id

        # Validate parent folder access if provided
        if folder_data.parent_id:
            try:
                await check_folder_access(
                    session, folder_data.parent_id, current_user, AccessMode.WRITE
                )
            except (PermissionError, ValueError) as e:
                raise HTTPException(status_code=403, detail=str(e))

        # Check for duplicate name in same parent
        stmt = select(Folder).where(
            Folder.name == folder_data.name,
            Folder.parent_id == folder_data.parent_id,
            Folder.created_by == user_id,
        )
        result = await session.execute(stmt)
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"Folder '{folder_data.name}' already exists in this location",
            )

        # Create folder
        folder = Folder(
            name=folder_data.name,
            description=folder_data.description,
            parent_id=folder_data.parent_id,
            created_by=user_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        session.add(folder)
        await session.commit()
        await session.refresh(folder)

        # Build response
        return FolderResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            parent_id=folder.parent_id,
            created_by=folder.created_by,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
            project_count=0,
            subfolder_count=0,
            is_owner=True,
            access_mode=AccessMode.ADMIN,
            is_shared=False,
        )


@router.get("/{folder_id}", response_model=FolderDetailResponse)
async def get_folder(
    folder_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Get folder details including breadcrumb."""
    async with db_session() as session:
        # Check access
        try:
            folder, is_owner, access_mode = await check_folder_access(
                session, folder_id, current_user, AccessMode.READ
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Get share count
        share_count_stmt = select(func.count(FolderShare.id)).where(
            FolderShare.folder_id == folder_id
        )
        share_count_result = await session.execute(share_count_stmt)
        share_count = share_count_result.scalar() or 0

        # Build breadcrumb
        breadcrumb = await _build_breadcrumb(session, folder_id)

        return FolderDetailResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            parent_id=folder.parent_id,
            created_by=folder.created_by,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
            is_owner=is_owner,
            access_mode=access_mode,
            share_count=share_count,
            breadcrumb=breadcrumb,
        )


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: uuid.UUID,
    folder_data: FolderUpdate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Update folder details.

    Requires ADMIN access. Can update name, description, and parent_id.
    Moving folder validates no circular references.
    """
    async with db_session() as session:
        # Check ADMIN access
        try:
            folder, is_owner, _ = await check_folder_access(
                session, folder_id, current_user, AccessMode.ADMIN
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Update name
        if folder_data.name is not None:
            # Check for duplicate
            stmt = select(Folder).where(
                Folder.name == folder_data.name,
                Folder.parent_id
                == (
                    folder_data.parent_id
                    if folder_data.parent_id is not None
                    else folder.parent_id
                ),
                Folder.created_by == folder.created_by,
                Folder.id != folder_id,
            )
            result = await session.execute(stmt)
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail=f"Folder '{folder_data.name}' already exists in this location",
                )
            folder.name = folder_data.name

        # Update description
        if folder_data.description is not None:
            folder.description = folder_data.description

        # Update parent (move folder)
        if folder_data.parent_id is not None:
            new_parent_id = folder_data.parent_id

            # Check for circular reference
            if await _check_circular_reference(session, folder_id, new_parent_id):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot move folder: would create circular reference",
                )

            # Validate access to new parent
            if new_parent_id is not None:
                try:
                    await check_folder_access(
                        session, new_parent_id, current_user, AccessMode.WRITE
                    )
                except (PermissionError, ValueError):
                    raise HTTPException(
                        status_code=403, detail="No access to target parent folder"
                    )

            folder.parent_id = new_parent_id

        folder.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(folder)

        # Build response
        project_count_stmt = select(func.count(Project.id)).where(
            Project.folder_id == folder_id
        )
        project_count_result = await session.execute(project_count_stmt)
        project_count = project_count_result.scalar() or 0

        subfolder_count_stmt = select(func.count(Folder.id)).where(
            Folder.parent_id == folder_id
        )
        subfolder_count_result = await session.execute(subfolder_count_stmt)
        subfolder_count = subfolder_count_result.scalar() or 0

        share_count_stmt = select(func.count(FolderShare.id)).where(
            FolderShare.folder_id == folder_id
        )
        share_count_result = await session.execute(share_count_stmt)
        is_shared = (share_count_result.scalar() or 0) > 0

        return FolderResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            parent_id=folder.parent_id,
            created_by=folder.created_by,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
            project_count=project_count,
            subfolder_count=subfolder_count,
            is_owner=is_owner,
            access_mode=AccessMode.ADMIN,
            is_shared=is_shared,
        )


@router.delete("/{folder_id}", status_code=204)
async def delete_folder(
    folder_id: uuid.UUID,
    move_to_root: bool = Query(
        False, description="Move projects and subfolders to root before deleting"
    ),
    delete_projects: bool = Query(
        False, description="Delete all projects in folder (dangerous)"
    ),
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Delete a folder.

    Requires ADMIN access.

    Must specify either move_to_root=true or delete_projects=true if folder has contents.
    """
    async with db_session() as session:
        # Check ADMIN access
        try:
            folder, _, _ = await check_folder_access(
                session, folder_id, current_user, AccessMode.ADMIN
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Check if folder has projects
        project_count_stmt = select(func.count(Project.id)).where(
            Project.folder_id == folder_id
        )
        project_count_result = await session.execute(project_count_stmt)
        project_count = project_count_result.scalar() or 0

        # Check if folder has subfolders
        subfolder_count_stmt = select(func.count(Folder.id)).where(
            Folder.parent_id == folder_id
        )
        subfolder_count_result = await session.execute(subfolder_count_stmt)
        subfolder_count = subfolder_count_result.scalar() or 0

        has_contents = project_count > 0 or subfolder_count > 0

        if has_contents and not move_to_root and not delete_projects:
            raise HTTPException(
                status_code=400,
                detail=f"Folder contains {project_count} projects and {subfolder_count} subfolders. "
                "Specify move_to_root=true or delete_projects=true",
            )

        if move_to_root and delete_projects:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both move_to_root and delete_projects",
            )

        # Handle contents
        if move_to_root:
            # Move projects to root
            await session.execute(
                select(Project)
                .where(Project.folder_id == folder_id)
                .execution_options(synchronize_session=False)
            )
            stmt = select(Project).where(Project.folder_id == folder_id)
            result = await session.execute(stmt)
            for project in result.scalars():
                project.folder_id = None

            # Move subfolders to root
            stmt = select(Folder).where(Folder.parent_id == folder_id)
            result = await session.execute(stmt)
            for subfolder in result.scalars():
                subfolder.parent_id = None

        elif delete_projects:
            # Delete all projects (shares will cascade)
            stmt = select(Project).where(Project.folder_id == folder_id)
            result = await session.execute(stmt)
            for project in result.scalars():
                await session.delete(project)

            # Move subfolders to root (don't delete them)
            stmt = select(Folder).where(Folder.parent_id == folder_id)
            result = await session.execute(stmt)
            for subfolder in result.scalars():
                subfolder.parent_id = None

        # Delete folder shares
        stmt = select(FolderShare).where(FolderShare.folder_id == folder_id)
        result = await session.execute(stmt)
        for share in result.scalars():
            await session.delete(share)

        # Delete folder
        await session.delete(folder)
        await session.commit()


@router.post("/{folder_id}/move", response_model=FolderResponse)
async def move_folder(
    folder_id: uuid.UUID,
    move_data: FolderMove,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Move folder to a new parent.

    Requires ADMIN on source folder and WRITE on target parent.
    Validates no circular references.
    """
    async with db_session() as session:
        # Check ADMIN access on source
        try:
            folder, is_owner, _ = await check_folder_access(
                session, folder_id, current_user, AccessMode.ADMIN
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        new_parent_id = move_data.new_parent_id

        # Check for circular reference
        if await _check_circular_reference(session, folder_id, new_parent_id):
            raise HTTPException(
                status_code=400,
                detail="Cannot move folder: would create circular reference",
            )

        # Validate access to new parent
        if new_parent_id is not None:
            try:
                await check_folder_access(
                    session, new_parent_id, current_user, AccessMode.WRITE
                )
            except (PermissionError, ValueError):
                raise HTTPException(
                    status_code=403, detail="No access to target parent folder"
                )

        folder.parent_id = new_parent_id
        folder.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(folder)

        # Build response
        project_count_stmt = select(func.count(Project.id)).where(
            Project.folder_id == folder_id
        )
        project_count_result = await session.execute(project_count_stmt)
        project_count = project_count_result.scalar() or 0

        subfolder_count_stmt = select(func.count(Folder.id)).where(
            Folder.parent_id == folder_id
        )
        subfolder_count_result = await session.execute(subfolder_count_stmt)
        subfolder_count = subfolder_count_result.scalar() or 0

        share_count_stmt = select(func.count(FolderShare.id)).where(
            FolderShare.folder_id == folder_id
        )
        share_count_result = await session.execute(share_count_stmt)
        is_shared = (share_count_result.scalar() or 0) > 0

        return FolderResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            parent_id=folder.parent_id,
            created_by=folder.created_by,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
            project_count=project_count,
            subfolder_count=subfolder_count,
            is_owner=is_owner,
            access_mode=AccessMode.ADMIN,
            is_shared=is_shared,
        )


# Sharing endpoints


@router.get("/{folder_id}/shares", response_model=t.List[FolderShareResponse])
async def list_folder_shares(
    folder_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """List all shares for a folder.

    Requires ADMIN access.
    """
    async with db_session() as session:
        # Check ADMIN access
        try:
            await check_folder_access(
                session, folder_id, current_user, AccessMode.ADMIN
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Get shares with user/team info
        stmt = (
            select(FolderShare)
            .where(FolderShare.folder_id == folder_id)
            .options(selectinload(FolderShare.user), selectinload(FolderShare.team))
        )
        result = await session.execute(stmt)
        shares = result.scalars().all()

        response = []
        for share in shares:
            response.append(
                FolderShareResponse(
                    id=share.id,
                    folder_id=share.folder_id,
                    share_type=share.share_type,
                    user_id=share.user_id,
                    team_id=share.team_id,
                    access_mode=share.access_mode,
                    created_at=share.created_at,
                    created_by=share.created_by,
                    user_email=share.user.email if share.user else None,
                    team_name=share.team.name if share.team else None,
                )
            )

        return response


@router.post("/{folder_id}/shares", response_model=FolderShareResponse, status_code=201)
async def create_folder_share(
    folder_id: uuid.UUID,
    share_data: FolderShareCreate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Create or update a folder share.

    Requires ADMIN access.
    If share already exists, updates the access_mode.
    """
    async with db_session() as session:
        # Check ADMIN access
        try:
            await check_folder_access(
                session, folder_id, current_user, AccessMode.ADMIN
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Validate user/team exists
        if share_data.share_type == ShareType.USER:
            stmt = select(User).where(User.id == share_data.user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
        else:
            stmt = select(Team).where(Team.id == share_data.team_id)
            result = await session.execute(stmt)
            team = result.scalar_one_or_none()
            if not team:
                raise HTTPException(status_code=404, detail="Team not found")

        # Check if share already exists
        stmt = select(FolderShare).where(
            FolderShare.folder_id == folder_id,
            FolderShare.share_type == share_data.share_type,
            (
                FolderShare.user_id == share_data.user_id
                if share_data.share_type == ShareType.USER
                else True
            ),
            (
                FolderShare.team_id == share_data.team_id
                if share_data.share_type == ShareType.TEAM
                else True
            ),
        )
        result = await session.execute(stmt)
        existing_share = result.scalar_one_or_none()

        if existing_share:
            # Update existing share
            existing_share.access_mode = share_data.access_mode
            share = existing_share
        else:
            # Create new share
            share = FolderShare(
                folder_id=folder_id,
                share_type=share_data.share_type,
                user_id=share_data.user_id,
                team_id=share_data.team_id,
                access_mode=share_data.access_mode,
                created_at=datetime.now(timezone.utc),
                created_by=current_user.user_id,
            )
            session.add(share)

        await session.commit()
        await session.refresh(share)

        # Load user/team for response
        stmt = (
            select(FolderShare)
            .where(FolderShare.id == share.id)
            .options(selectinload(FolderShare.user), selectinload(FolderShare.team))
        )
        result = await session.execute(stmt)
        share = result.scalar_one()

        return FolderShareResponse(
            id=share.id,
            folder_id=share.folder_id,
            share_type=share.share_type,
            user_id=share.user_id,
            team_id=share.team_id,
            access_mode=share.access_mode,
            created_at=share.created_at,
            created_by=share.created_by,
            user_email=share.user.email if share.user else None,
            team_name=share.team.name if share.team else None,
        )


@router.delete("/{folder_id}/shares/{share_id}", status_code=204)
async def delete_folder_share(
    folder_id: uuid.UUID,
    share_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Remove a folder share.

    Requires ADMIN access.
    """
    async with db_session() as session:
        # Check ADMIN access
        try:
            await check_folder_access(
                session, folder_id, current_user, AccessMode.ADMIN
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Get share
        stmt = select(FolderShare).where(
            FolderShare.id == share_id,
            FolderShare.folder_id == folder_id,
        )
        result = await session.execute(stmt)
        share = result.scalar_one_or_none()

        if not share:
            raise HTTPException(status_code=404, detail="Share not found")

        await session.delete(share)
        await session.commit()


@router.get("/{folder_id}/effective-permissions", response_model=EffectivePermissions)
async def get_folder_effective_permissions(
    folder_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Get user's effective permissions for a folder.

    Shows all permission sources and final resolved access mode.
    Useful for debugging permission inheritance.
    """
    async with db_session() as session:
        try:
            is_owner, access_mode, permissions = await get_effective_folder_permissions(
                session, folder_id, current_user
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))

        permission_sources = [
            PermissionSourceDetail(
                source_type=p.source,
                source_id=p.source_id,
                source_name=p.source_name,
                access_mode=p.access_mode,
                folder_path=p.folder_path,
            )
            for p in permissions
        ]

        return EffectivePermissions(
            access_mode=access_mode,
            is_owner=is_owner,
            permission_sources=permission_sources,
        )
