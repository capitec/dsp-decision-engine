import typing as t
import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    Column,
    UniqueConstraint,
    ForeignKey,
    Index,
    CheckConstraint,
    Integer,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP, ARRAY
from sqlmodel import SQLModel, Field, Relationship
from pydantic import field_serializer, field_validator
from .users import User, Team

# Forward reference for type hints
if t.TYPE_CHECKING:
    from .folders import Folder


class ProjectType(str, Enum):
    FLAT_RULE = "flat_rule"
    TREE = "tree"


class ShareType(str, Enum):
    USER = "user"
    TEAM = "team"


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class RevisionType(str, Enum):
    MANUAL = "manual"
    AUTO_SAVE = "auto_save"
    DEPLOYMENT = "deployment"


class Project(SQLModel, table=True):
    __tablename__ = "projects"
    __table_args__ = (
        Index("ix_projects_folder_id", "folder_id"),
        {"schema": "app_schema"},
    )

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    name: str
    description: t.Optional[str] = None
    project_type: ProjectType = Field(
        sa_column=Column(
            SQLEnum(
                ProjectType,
                schema="app_schema",
                values_callable=lambda x: [e.value for e in x],
            )
        )
    )
    folder_id: t.Optional[uuid.UUID] = Field(
        default=None,
        sa_column=Column(UUID, ForeignKey("app_schema.folders.id"), nullable=True),
    )
    created_by: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.users.id"))
    )
    # Store as dict in DB but validate/serialize as TTree
    project_data: t.Dict[str, t.Any] = Field(
        sa_column=Column(JSONB),
        description="Complete project state - dict representation of TTree (UiTree or PrioritizedTreeModule)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    version: int = Field(
        default=1,
        sa_column=Column(Integer, nullable=False, server_default="1"),
        description="Bumped by every save of project_data; a save naming an older base_version is refused",
    )

    # Relationships
    folder: t.Optional["Folder"] = Relationship(
        back_populates="projects",
        sa_relationship_kwargs={"foreign_keys": "Project.folder_id"},
    )
    created_by_user: User = Relationship(
        back_populates="created_projects",
        sa_relationship_kwargs={"foreign_keys": "Project.created_by"},
    )
    shares: t.List["ProjectShare"] = Relationship(back_populates="project")


class ProjectOp(SQLModel, table=True):
    """One project version: the ops that made it from the one before, and who sent them."""

    __tablename__ = "project_ops"
    __table_args__ = (
        # A message resent by a client that never saw its ack is logged once.
        Index("uq_project_ops_message", "project_id", "client_id", "msg_id", unique=True),
        {"schema": "app_schema"},
    )

    project_id: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.projects.id", ondelete="CASCADE"), primary_key=True)
    )
    version: int = Field(sa_column=Column(Integer, primary_key=True))
    client_id: str = Field(sa_column=Column(Text, nullable=False))
    msg_id: str = Field(sa_column=Column(Text, nullable=False))
    ops: t.List[t.Dict[str, t.Any]] = Field(sa_column=Column(JSONB, nullable=False))
    # The ids of the earlier messages an offline batch contains: one of them delivered late is not applied again.
    parts: t.List[str] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, server_default="[]"))
    # The whole document at this version, stored every few versions so a rebuild replays only the ones after it
    # (SQL NULL otherwise: JSON null would make every row look like one).
    snapshot: t.Optional[t.Dict[str, t.Any]] = Field(
        default=None, sa_column=Column(JSONB(none_as_null=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True), nullable=False),
    )


class ProjectShare(SQLModel, table=True):
    __tablename__ = "project_shares"
    __table_args__ = {"schema": "app_schema"}

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    project_id: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.projects.id"))
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
    project: Project = Relationship(back_populates="shares")
    user: t.Optional[User] = Relationship(
        back_populates="project_shares",
        sa_relationship_kwargs={"foreign_keys": "ProjectShare.user_id"},
    )
    team: t.Optional[Team] = Relationship(back_populates="project_shares")
    created_by_user: User = Relationship(
        sa_relationship_kwargs={"foreign_keys": "ProjectShare.created_by"}
    )
