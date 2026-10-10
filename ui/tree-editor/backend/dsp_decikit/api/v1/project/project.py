import typing as t
import uuid
from datetime import datetime, timezone
import json
import re
from urllib.parse import quote

import decikit

from fastapi import Depends, HTTPException, Query, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select, case, update
from fastapi_pagination import Page, Params, paginate
from pydantic import BaseModel, ConfigDict, Field

from dsp_decikit.db.session import db_session
from dsp_decikit.db.models.projects import (
    Project,
    ProjectShare,
    ShareType,
    AccessMode,
    ProjectType,
)
from dsp_decikit.db.models.users import Team, User
from dsp_decikit.auth.authorization import (
    get_current_user_session,
    is_superadmin,
    get_user_teams,
)
from dsp_decikit.auth.models import SessionInfo
from .routes import router

from dsp_decikit.services.documents import (
    check_document,
    default_project_data,
    detect_project_type,
    imported_document,
    validation_errors,
)

# ================================
# Request/Response Models
# ================================


class ProjectCreate(BaseModel):
    name: str
    description: t.Optional[str] = None
    project_type: ProjectType = ProjectType.TREE
    folder_id: t.Optional[uuid.UUID] = None
    project_data: t.Optional[t.Dict[str, t.Any]] = None


class ProjectUpdate(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None
    folder_id: t.Optional[uuid.UUID] = None
    project_data: t.Optional[t.Dict[str, t.Any]] = None
    base_version: t.Optional[int] = Field(
        None,
        description="The version the caller last saw; a save is refused (409) when the project has moved on. Omit to overwrite.",
    )


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: t.Optional[str]
    project_type: ProjectType
    folder_id: t.Optional[uuid.UUID]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    version: int
    is_owner: bool
    access_mode: AccessMode


class ProjectCreateResponse(ProjectResponse):
    validation_errors: t.List[t.Dict[str, t.Any]]


class ProjectDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: t.Optional[str]
    project_type: ProjectType
    folder_id: t.Optional[uuid.UUID]
    project_data: t.Dict[str, t.Any]
    validation_errors: t.List[t.Dict[str, t.Any]]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    version: int
    is_owner: bool
    access_mode: AccessMode

    @classmethod
    def from_project(cls, project: Project, is_owner: bool, access_mode: AccessMode):
        return cls(
            id=project.id,
            name=project.name,
            description=project.description,
            project_type=project.project_type,
            folder_id=project.folder_id,
            project_data=project.project_data,
            validation_errors=validation_errors(project.project_data),
            created_by=project.created_by,
            created_at=project.created_at,
            updated_at=project.updated_at,
            version=project.version,
            is_owner=is_owner,
            access_mode=access_mode,
        )


class ShareProjectRequest(BaseModel):
    share_type: ShareType
    user_id: t.Optional[uuid.UUID] = None
    team_id: t.Optional[uuid.UUID] = None
    access_mode: AccessMode


class ProjectShareResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    share_type: ShareType
    user_id: t.Optional[uuid.UUID]
    team_id: t.Optional[uuid.UUID]
    access_mode: AccessMode
    created_at: datetime
    created_by: uuid.UUID
    user_email: t.Optional[str] = None
    team_name: t.Optional[str] = None


class ProjectImportResponse(BaseModel):
    project_id: uuid.UUID
    project_name: str
    project_type: ProjectType
    validation_errors: t.List[t.Dict[str, t.Any]]


class TreeVerificationResponse(BaseModel):
    valid: bool
    errors: t.List[str]
    warnings: t.List[str]
    node_count: int
    feature_count: int
    subtree_count: t.Optional[int]
    format_version: t.Optional[int]
    node_results: t.List[t.Dict[str, t.Any]]


# ================================
# Helper Functions
# ================================


async def _check_project_access(
    project_id: uuid.UUID, current_user: SessionInfo
) -> tuple[Project, bool, AccessMode]:
    """Check project access and return project, is_owner, access_mode.

    Now uses folder_permissions service for hierarchical permission resolution.
    """
    from dsp_decikit.services.folder_permissions import check_project_access

    async with db_session() as session:
        try:
            project, is_owner, access_mode = await check_project_access(
                session, project_id, current_user, AccessMode.READ
            )
            return project, is_owner, access_mode
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


# ================================
# Project CRUD Endpoints
# ================================


def _newest_first(projects: t.List[ProjectResponse]) -> t.List[ProjectResponse]:
    # Paging slices this list, so its order must not depend on how Postgres happens to return rows.
    return sorted(projects, key=lambda p: (p.updated_at, p.id), reverse=True)


@router.get("/projects", response_model=Page[ProjectResponse])
async def list_projects(
    current_user: SessionInfo = Depends(get_current_user_session),
    params: Params = Depends(),
    search: t.Optional[str] = Query(
        None, description="Search projects by name or description"
    ),
    project_type: t.Optional[ProjectType] = Query(
        None, description="Filter by project type"
    ),
    folder_id: t.Optional[uuid.UUID] = Query(
        None, description="Filter by folder (null for root projects)"
    ),
):
    """List all projects accessible to the current user"""
    async with db_session() as session:
        if await is_superadmin(current_user):
            # Superadmins see all projects
            query = select(Project)
            if search:
                query = query.where(
                    (Project.name.icontains(search, autoescape=True))
                    | (Project.description.icontains(search, autoescape=True))
                )
            if project_type:
                query = query.where(Project.project_type == project_type)
            if folder_id is not None:
                query = query.where(Project.folder_id == folder_id)

            result = await session.execute(query)
            all_projects = result.scalars().all()

            project_responses = []
            for project in all_projects:
                is_owner = project.created_by == current_user.user_id
                project_responses.append(
                    ProjectResponse(
                        **project.__dict__,
                        is_owner=is_owner,
                        access_mode=AccessMode.ADMIN,
                    )
                )

            return paginate(_newest_first(project_responses), params)

        # Get user's teams for shared project lookup
        user_teams = await get_user_teams(current_user)

        # Get projects owned by user
        owned_query = select(Project).where(Project.created_by == current_user.user_id)
        if search:
            owned_query = owned_query.where(
                (Project.name.icontains(search, autoescape=True))
                | (Project.description.icontains(search, autoescape=True))
            )
        if project_type:
            owned_query = owned_query.where(Project.project_type == project_type)
        if folder_id is not None:
            owned_query = owned_query.where(Project.folder_id == folder_id)

        owned_result = await session.execute(owned_query)
        owned_projects = owned_result.scalars().all()

        # Get projects shared with user directly
        shared_query = (
            select(Project, ProjectShare.access_mode)
            .join(ProjectShare)
            .where(ProjectShare.user_id == current_user.user_id)
        )
        if search:
            shared_query = shared_query.where(
                (Project.name.icontains(search, autoescape=True))
                | (Project.description.icontains(search, autoescape=True))
            )
        if project_type:
            shared_query = shared_query.where(Project.project_type == project_type)
        if folder_id is not None:
            shared_query = shared_query.where(Project.folder_id == folder_id)

        shared_result = await session.execute(shared_query)
        shared_projects = shared_result.all()

        # Get projects shared with user's teams
        team_shared_query = (
            select(Project, ProjectShare.access_mode)
            .join(ProjectShare)
            .join(Team, ProjectShare.team_id == Team.id)
            .where(Team.name.in_(user_teams))
        )
        if search:
            team_shared_query = team_shared_query.where(
                (Project.name.icontains(search, autoescape=True))
                | (Project.description.icontains(search, autoescape=True))
            )
        if project_type:
            team_shared_query = team_shared_query.where(
                Project.project_type == project_type
            )
        if folder_id is not None:
            team_shared_query = team_shared_query.where(Project.folder_id == folder_id)

        team_shared_result = await session.execute(team_shared_query)
        team_shared_projects = team_shared_result.all()

        # Combine results
        project_responses = []

        # Add owned projects
        for project in owned_projects:
            project_responses.append(
                ProjectResponse(
                    **project.__dict__, is_owner=True, access_mode=AccessMode.ADMIN
                )
            )

        # Add shared projects (avoid duplicates)
        seen_project_ids = {p.id for p in owned_projects}

        for project, access_mode in shared_projects + team_shared_projects:
            if project.id not in seen_project_ids:
                project_responses.append(
                    ProjectResponse(
                        **project.__dict__, is_owner=False, access_mode=access_mode
                    )
                )
                seen_project_ids.add(project.id)

        return paginate(_newest_first(project_responses), params)


@router.post("/projects", response_model=ProjectCreateResponse)
async def create_project(
    project_data: ProjectCreate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Create a new project"""
    async with db_session() as session:
        # Validate folder access if folder_id provided
        if project_data.folder_id:
            from dsp_decikit.services.folder_permissions import check_folder_access

            try:
                await check_folder_access(
                    session, project_data.folder_id, current_user, AccessMode.WRITE
                )
            except (PermissionError, ValueError) as e:
                raise HTTPException(
                    status_code=403, detail=f"Folder access denied: {e}"
                )

        data = project_data.project_data
        if data is None:
            data = default_project_data(project_data.project_type)
        try:
            check_document(data, project_data.project_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        project = Project(
            name=project_data.name,
            description=project_data.description,
            project_type=project_data.project_type,
            folder_id=project_data.folder_id,
            project_data=data,
            created_by=current_user.user_id,
        )

        session.add(project)
        await session.commit()
        await session.refresh(project)

        return ProjectCreateResponse(
            **project.__dict__,
            is_owner=True,
            access_mode=AccessMode.ADMIN,
            validation_errors=validation_errors(data),
        )


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Get project details. project_data is returned as stored; validation_errors lists any problems."""
    project, is_owner, access_mode = await _check_project_access(
        project_id, current_user
    )
    return ProjectDetailResponse.from_project(project, is_owner, access_mode)


@router.put("/projects/{project_id}", response_model=ProjectDetailResponse)
async def update_project(
    project_id: uuid.UUID,
    project_data: ProjectUpdate,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Update project (metadata and/or project_data)"""
    project, is_owner, access_mode = await _check_project_access(
        project_id, current_user
    )

    # Check write permissions
    if access_mode not in [AccessMode.WRITE, AccessMode.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    async with db_session() as session:
        # Reload project in this session
        query = select(Project).where(Project.id == project_id)
        result = await session.execute(query)
        project = result.scalar_one()

        changes: t.Dict[str, t.Any] = {}
        if project_data.name is not None:
            changes["name"] = project_data.name
        if project_data.description is not None:
            changes["description"] = project_data.description
        if project_data.folder_id is not None:
            # Validate access to new folder
            from dsp_decikit.services.folder_permissions import check_folder_access

            try:
                await check_folder_access(
                    session, project_data.folder_id, current_user, AccessMode.WRITE
                )
            except (PermissionError, ValueError) as e:
                raise HTTPException(
                    status_code=403, detail=f"Target folder access denied: {e}"
                )
            changes["folder_id"] = project_data.folder_id
        if project_data.project_data is not None:
            try:
                check_document(project_data.project_data, project.project_type)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            changes["project_data"] = project_data.project_data

        # One conditional statement, so two saves of the same base cannot both pass the check
        stmt = update(Project).where(Project.id == project_id)
        if project_data.base_version is not None:
            stmt = stmt.where(Project.version == project_data.base_version)
        saved = await session.execute(
            stmt.values(
                **changes, version=Project.version + 1, updated_at=datetime.now(timezone.utc)
            )
            .returning(Project.version)
            .execution_options(synchronize_session=False)
        )
        if saved.first() is None:
            await session.refresh(project)
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "stale_version",
                    "message": "The project was saved by someone else since you opened it",
                    "current_version": project.version,
                },
            )

        await session.commit()
        await session.refresh(project)

        return ProjectDetailResponse.from_project(project, is_owner, access_mode)


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Delete a project"""
    _, is_owner, _ = await _check_project_access(project_id, current_user)

    # Only owners and superadmins can delete projects
    if not is_owner and not await is_superadmin(current_user):
        raise HTTPException(
            status_code=403,
            detail="Only project owners and superadmins can delete projects",
        )

    async with db_session() as session:
        query = select(Project).where(Project.id == project_id)
        result = await session.execute(query)
        project = result.scalar_one()

        await session.delete(project)
        await session.commit()

        return {"message": "Project deleted successfully"}


class ProjectMove(BaseModel):
    """Schema for moving a project to a folder."""

    new_folder_id: t.Optional[uuid.UUID] = Field(
        None, description="New folder ID (null for root)"
    )


@router.post("/projects/{project_id}/move", response_model=ProjectResponse)
async def move_project(
    project_id: uuid.UUID,
    move_data: ProjectMove,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Move project to a different folder.

    Requires WRITE access on project and WRITE access on target folder (if not null).
    """
    async with db_session() as session:
        # Check WRITE access on project
        from dsp_decikit.services.folder_permissions import check_project_access

        try:
            project, is_owner, access_mode = await check_project_access(
                session, project_id, current_user, AccessMode.WRITE
            )
        except (PermissionError, ValueError) as e:
            raise HTTPException(status_code=403, detail=str(e))

        # Validate access to new folder if provided
        if move_data.new_folder_id is not None:
            from dsp_decikit.services.folder_permissions import check_folder_access

            try:
                await check_folder_access(
                    session, move_data.new_folder_id, current_user, AccessMode.WRITE
                )
            except (PermissionError, ValueError) as e:
                raise HTTPException(
                    status_code=403, detail=f"Target folder access denied: {e}"
                )

        # Move project
        project.folder_id = move_data.new_folder_id
        project.version = Project.version + 1
        project.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(project)

        return ProjectResponse(
            **project.__dict__,
            is_owner=is_owner,
            access_mode=access_mode,
        )


# ================================
# Sharing Endpoints
# ================================


@router.get(
    "/projects/{project_id}/shares", response_model=t.List[ProjectShareResponse]
)
async def get_project_shares(
    project_id: uuid.UUID, current_user: SessionInfo = Depends(get_current_user_session)
):
    """Get all shares for a project"""
    _, is_owner, access_mode = await _check_project_access(project_id, current_user)

    # Only owners and admins can view shares
    if not is_owner and access_mode != AccessMode.ADMIN:
        raise HTTPException(
            status_code=403, detail="Only project owners and admins can view shares"
        )

    async with db_session() as session:
        query = (
            select(ProjectShare, User.email, Team.name)
            .outerjoin(User, ProjectShare.user_id == User.id)
            .outerjoin(Team, ProjectShare.team_id == Team.id)
            .where(ProjectShare.project_id == project_id)
        )
        result = await session.execute(query)
        shares_data = result.all()

        shares = []
        for share, user_email, team_name in shares_data:
            shares.append(
                ProjectShareResponse(
                    **share.__dict__, user_email=user_email, team_name=team_name
                )
            )

        return shares


@router.post("/projects/{project_id}/shares", response_model=ProjectShareResponse)
async def share_project(
    project_id: uuid.UUID,
    share_request: ShareProjectRequest,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Share a project with a user or team"""
    _, is_owner, access_mode = await _check_project_access(project_id, current_user)

    # Only owners and admins can share projects
    if not is_owner and access_mode != AccessMode.ADMIN:
        raise HTTPException(
            status_code=403, detail="Only project owners and admins can share projects"
        )

    # Validate share request
    if share_request.share_type == ShareType.USER and not share_request.user_id:
        raise HTTPException(status_code=400, detail="user_id required for user shares")
    if share_request.share_type == ShareType.TEAM and not share_request.team_id:
        raise HTTPException(status_code=400, detail="team_id required for team shares")

    async with db_session() as session:
        # Check if share already exists
        existing_query = select(ProjectShare).where(
            ProjectShare.project_id == project_id,
            ProjectShare.share_type == share_request.share_type,
            ProjectShare.user_id == share_request.user_id,
            ProjectShare.team_id == share_request.team_id,
        )
        existing_result = await session.execute(existing_query)
        existing_share = existing_result.scalar_one_or_none()

        if existing_share:
            # Update existing share
            existing_share.access_mode = share_request.access_mode
            await session.commit()
            await session.refresh(existing_share)
            share = existing_share
        else:
            # Create new share
            share = ProjectShare(
                project_id=project_id,
                share_type=share_request.share_type,
                user_id=share_request.user_id,
                team_id=share_request.team_id,
                access_mode=share_request.access_mode,
                created_by=current_user.user_id,
            )
            session.add(share)
            await session.commit()
            await session.refresh(share)

        # Get user/team details for response
        user_email = None
        team_name = None

        if share.user_id:
            user_query = select(User.email).where(User.id == share.user_id)
            user_result = await session.execute(user_query)
            user_email = user_result.scalar_one_or_none()

        if share.team_id:
            team_query = select(Team.name).where(Team.id == share.team_id)
            team_result = await session.execute(team_query)
            team_name = team_result.scalar_one_or_none()

        return ProjectShareResponse(
            **share.__dict__, user_email=user_email, team_name=team_name
        )


@router.delete("/projects/{project_id}/shares/{share_id}")
async def remove_project_share(
    project_id: uuid.UUID,
    share_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Remove a project share"""
    _, is_owner, access_mode = await _check_project_access(project_id, current_user)

    # Only owners and admins can remove shares
    if not is_owner and access_mode != AccessMode.ADMIN:
        raise HTTPException(
            status_code=403, detail="Only project owners and admins can remove shares"
        )

    async with db_session() as session:
        query = select(ProjectShare).where(
            ProjectShare.id == share_id, ProjectShare.project_id == project_id
        )
        result = await session.execute(query)
        share = result.scalar_one_or_none()

        if not share:
            raise HTTPException(status_code=404, detail="Share not found")

        await session.delete(share)
        await session.commit()

        return {"message": "Share removed successfully"}


# ================================
# User/Team Search
# ================================


@router.get("/users/search")
async def search_users(q: str, _: SessionInfo = Depends(get_current_user_session)):
    """Search users by email for sharing projects"""
    if len(q) < 2:
        return {"users": []}

    async with db_session() as session:
        query = (
            select(User.id, User.email, User.full_name)
            .where(User.email.icontains(q, autoescape=True))
            .where(User.is_active == True)
            .limit(10)
        )
        result = await session.execute(query)
        users = result.all()

        return {
            "users": [
                {"id": str(user.id), "email": user.email, "full_name": user.full_name}
                for user in users
            ]
        }


@router.get("/teams/search")
async def search_teams(q: str, _: SessionInfo = Depends(get_current_user_session)):
    """Search teams by name for sharing projects"""
    if len(q) < 2:
        return {"teams": []}

    async with db_session() as session:
        query = (
            select(Team.id, Team.name, Team.description)
            .where(Team.name.icontains(q, autoescape=True))
            .limit(10)
        )
        result = await session.execute(query)
        teams = result.all()

        return {
            "teams": [
                {"id": str(team.id), "name": team.name, "description": team.description}
                for team in teams
            ]
        }


# ================================
# Export/Import Endpoints
# ================================


@router.get("/projects/{project_id}/export")
async def export_project(
    project_id: uuid.UUID,
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Export project as a JSON file. Fails with 400 if the document is invalid."""
    project, _, _ = await _check_project_access(project_id, current_user)

    errors = validation_errors(project.project_data)
    if errors:
        return JSONResponse(
            status_code=400,
            content={
                "message": "Project is invalid and cannot be exported",
                "validation_errors": errors,
            },
        )

    export_data = dict(project.project_data)
    # The document's own metadata (its name, description) wins over the project's name; the stamp is fresh.
    export_data["metadata"] = {
        "name": project.name,
        **(export_data.get("metadata") or {}),
        "project_id": str(project.id),
        "project_type": project.project_type.value,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "decikit_version": decikit.__version__,
    }

    formatted_json = json.dumps(export_data, indent=2, ensure_ascii=False)
    # Any name may hold quotes or non-Latin-1 characters: an ASCII fallback plus the RFC 5987 UTF-8 form
    filename = project.name.replace(" ", "_") + "_export.json"
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return Response(
        content=formatted_json,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"
        },
    )


@router.post("/projects/import", response_model=ProjectImportResponse)
async def import_project(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    current_user: SessionInfo = Depends(get_current_user_session),
):
    """Import project from JSON file - auto-detects type from structure"""
    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    # Read and parse JSON
    try:
        content = await file.read()
        project_data_dict = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    project_data_dict = imported_document(project_data_dict)
    try:
        detected_type = detect_project_type(project_data_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create project with auto-detected type
    async with db_session() as session:
        new_project = Project(
            name=name,
            description=description,
            project_type=detected_type,
            project_data=project_data_dict,
            created_by=current_user.user_id,
        )

        session.add(new_project)
        await session.commit()
        await session.refresh(new_project)

        return ProjectImportResponse(
            project_id=new_project.id,
            project_name=new_project.name,
            project_type=new_project.project_type,
            validation_errors=validation_errors(project_data_dict),
        )
