import typing as t
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    UniqueConstraint,
    ForeignKey,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlmodel import SQLModel, Field, Relationship

from .users import User, Team
from .projects import ShareType, AccessMode

# Forward reference for type hints
if t.TYPE_CHECKING:
    from .projects import Project


class Folder(SQLModel, table=True):
    __tablename__ = "folders"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "parent_id",
            "created_by",
            name="uq_folder_name_parent_creator",
        ),
        Index("ix_folders_parent_id", "parent_id"),
        Index("ix_folders_created_by", "created_by"),
        Index("ix_folders_name", "name"),
        {"schema": "app_schema"},
    )

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    name: str = Field(max_length=255)
    description: t.Optional[str] = None
    parent_id: t.Optional[uuid.UUID] = Field(
        default=None,
        sa_column=Column(UUID, ForeignKey("app_schema.folders.id"), nullable=True),
    )
    created_by: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.users.id"))
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )

    # Relationships
    parent: t.Optional["Folder"] = Relationship(
        back_populates="children",
        sa_relationship_kwargs={
            "remote_side": "Folder.id",
            "foreign_keys": "Folder.parent_id",
        },
    )
    children: t.List["Folder"] = Relationship(
        back_populates="parent",
        sa_relationship_kwargs={
            "foreign_keys": "Folder.parent_id",
        },
    )
    projects: t.List["Project"] = Relationship(
        back_populates="folder",
        sa_relationship_kwargs={"foreign_keys": "Project.folder_id"},
    )
    created_by_user: User = Relationship(
        back_populates="created_folders",
        sa_relationship_kwargs={"foreign_keys": "Folder.created_by"},
    )
    shares: t.List["FolderShare"] = Relationship(back_populates="folder")


class FolderShare(SQLModel, table=True):
    __tablename__ = "folder_shares"
    __table_args__ = (
        UniqueConstraint(
            "folder_id",
            "share_type",
            "user_id",
            "team_id",
            name="uq_folder_share_unique",
        ),
        Index("ix_folder_shares_folder_id", "folder_id"),
        Index("ix_folder_shares_user_id", "user_id"),
        Index("ix_folder_shares_team_id", "team_id"),
        {"schema": "app_schema"},
    )

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    folder_id: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.folders.id"))
    )
    share_type: ShareType = Field(
        sa_column=Column(SQLEnum(ShareType, schema="app_schema"))
    )
    user_id: t.Optional[uuid.UUID] = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.users.id")), default=None
    )
    team_id: t.Optional[uuid.UUID] = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.teams.id")), default=None
    )
    access_mode: AccessMode = Field(
        sa_column=Column(SQLEnum(AccessMode, schema="app_schema"))
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    created_by: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.users.id"))
    )

    # Relationships
    folder: Folder = Relationship(back_populates="shares")
    user: t.Optional[User] = Relationship(
        back_populates="folder_shares",
        sa_relationship_kwargs={"foreign_keys": "FolderShare.user_id"},
    )
    team: t.Optional[Team] = Relationship(back_populates="folder_shares")
    created_by_user: User = Relationship(
        sa_relationship_kwargs={"foreign_keys": "FolderShare.created_by"}
    )
