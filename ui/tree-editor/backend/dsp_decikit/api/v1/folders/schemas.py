"""Pydantic schemas for folder API endpoints."""

import typing as t
import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from dsp_decikit.db.models.projects import AccessMode, ShareType
from dsp_decikit.services.folder_permissions import PermissionSource

# Request schemas


class FolderCreate(BaseModel):
    """Schema for creating a new folder."""

    name: str = Field(..., min_length=1, max_length=255, description="Folder name")
    description: t.Optional[str] = Field(
        None, description="Optional folder description"
    )
    parent_id: t.Optional[uuid.UUID] = Field(
        None, description="Parent folder ID (null for root)"
    )


class FolderUpdate(BaseModel):
    """Schema for updating an existing folder."""

    name: t.Optional[str] = Field(
        None, min_length=1, max_length=255, description="Folder name"
    )
    description: t.Optional[str] = Field(None, description="Folder description")
    parent_id: t.Optional[uuid.UUID] = Field(
        None, description="Parent folder ID (use 'null' string to move to root)"
    )


class FolderMove(BaseModel):
    """Schema for moving a folder to a new parent."""

    new_parent_id: t.Optional[uuid.UUID] = Field(
        None, description="New parent folder ID (null for root)"
    )


class FolderShareCreate(BaseModel):
    """Schema for creating/updating a folder share."""

    share_type: ShareType = Field(..., description="Share with user or team")
    user_id: t.Optional[uuid.UUID] = Field(
        None, description="User ID (required if share_type=USER)"
    )
    team_id: t.Optional[uuid.UUID] = Field(
        None, description="Team ID (required if share_type=TEAM)"
    )
    access_mode: AccessMode = Field(
        ..., description="Access level: READ, WRITE, or ADMIN"
    )

    def model_post_init(self, __context: t.Any) -> None:
        """Validate that exactly one of user_id or team_id is set."""
        if self.share_type == ShareType.USER and self.user_id is None:
            raise ValueError("user_id is required when share_type is USER")
        if self.share_type == ShareType.TEAM and self.team_id is None:
            raise ValueError("team_id is required when share_type is TEAM")
        if self.share_type == ShareType.USER and self.team_id is not None:
            raise ValueError("team_id must be null when share_type is USER")
        if self.share_type == ShareType.TEAM and self.user_id is not None:
            raise ValueError("user_id must be null when share_type is TEAM")


# Response schemas


class FolderBreadcrumb(BaseModel):
    """Breadcrumb item for folder path."""

    id: uuid.UUID
    name: str


class FolderResponse(BaseModel):
    """Basic folder information."""

    id: uuid.UUID
    name: str
    description: t.Optional[str]
    parent_id: t.Optional[uuid.UUID]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime

    # Computed fields
    project_count: int = Field(
        ..., description="Number of projects directly in this folder"
    )
    subfolder_count: int = Field(..., description="Number of immediate subfolders")
    is_owner: bool = Field(..., description="True if current user owns this folder")
    access_mode: AccessMode = Field(..., description="User's effective access level")
    is_shared: bool = Field(..., description="True if folder has any shares")

    model_config = {"from_attributes": True}


class FolderDetailResponse(BaseModel):
    """Detailed folder information with contents."""

    id: uuid.UUID
    name: str
    description: t.Optional[str]
    parent_id: t.Optional[uuid.UUID]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime

    # Computed fields
    is_owner: bool
    access_mode: AccessMode
    share_count: int = Field(..., description="Total number of shares on this folder")
    breadcrumb: t.List[FolderBreadcrumb] = Field(
        ..., description="Path from root to this folder"
    )

    model_config = {"from_attributes": True}


class FolderShareResponse(BaseModel):
    """Folder share information."""

    id: uuid.UUID
    folder_id: uuid.UUID
    share_type: ShareType
    user_id: t.Optional[uuid.UUID]
    team_id: t.Optional[uuid.UUID]
    access_mode: AccessMode
    created_at: datetime
    created_by: uuid.UUID

    # Populated fields
    user_email: t.Optional[str] = Field(
        None, description="Email of shared user (if USER share)"
    )
    team_name: t.Optional[str] = Field(
        None, description="Name of shared team (if TEAM share)"
    )

    model_config = {"from_attributes": True}


class PermissionSourceDetail(BaseModel):
    """Details about a permission source."""

    source_type: PermissionSource
    source_id: t.Optional[uuid.UUID] = None
    source_name: t.Optional[str] = None
    access_mode: AccessMode
    folder_path: t.Optional[str] = None


class EffectivePermissions(BaseModel):
    """Effective permissions and their sources."""

    access_mode: AccessMode = Field(
        ..., description="User's final effective access mode"
    )
    is_owner: bool = Field(..., description="True if user is the owner")
    permission_sources: t.List[PermissionSourceDetail] = Field(
        ..., description="All permission sources that contributed to final access"
    )
