import typing as t
import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    UniqueConstraint,
    ForeignKey,
    LargeBinary,
    UUID,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, TIMESTAMP
from sqlmodel import SQLModel, Field, Relationship
from dsp_decider.auth.settings import _ApiKeyAuthorizerSettings
from dsp_decider.auth.models import TeamRole, TokenScope

# Forward references for type hints
if t.TYPE_CHECKING:
    from .folders import Folder, FolderShare


class User(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = {"schema": "app_schema"}

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    email: str = Field(unique=True, index=True)  # Always stored as lowercase
    full_name: t.Optional[str] = None
    password_hash: t.Optional[str] = None  # For basic auth
    profile_picture: t.Optional[bytes] = Field(
        default=None, sa_column=Column(LargeBinary)
    )  # Store image as bytes
    is_active: bool = Field(default=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )

    # Relationships
    created_projects: t.List["Project"] = Relationship(
        back_populates="created_by_user",
        sa_relationship_kwargs={"foreign_keys": "Project.created_by"},
    )
    project_shares: t.List["ProjectShare"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"foreign_keys": "ProjectShare.user_id"},
    )
    created_folders: t.List["Folder"] = Relationship(
        back_populates="created_by_user",
        sa_relationship_kwargs={"foreign_keys": "Folder.created_by"},
    )
    folder_shares: t.List["FolderShare"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"foreign_keys": "FolderShare.user_id"},
    )
    team_memberships: t.List["TeamMembership"] = Relationship(back_populates="user")
    created_teams: t.List["Team"] = Relationship(
        back_populates="created_by_user",
        sa_relationship_kwargs={"foreign_keys": "Team.created_by"},
    )
    api_tokens: t.List["ApiToken"] = Relationship(back_populates="user")


class Team(SQLModel, table=True):
    __tablename__ = "teams"
    __table_args__ = {"schema": "app_schema"}

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    name: str = Field(unique=True, index=True)
    description: t.Optional[str] = None
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
    members: t.List["TeamMembership"] = Relationship(back_populates="team")
    project_shares: t.List["ProjectShare"] = Relationship(back_populates="team")
    folder_shares: t.List["FolderShare"] = Relationship(back_populates="team")
    created_by_user: "User" = Relationship(
        back_populates="created_teams",
        sa_relationship_kwargs={"foreign_keys": "Team.created_by"},
    )


class TeamMembership(SQLModel, table=True):
    __tablename__ = "team_memberships"
    __table_args__ = (UniqueConstraint("user_id", "team_id"), {"schema": "app_schema"})

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    user_id: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.users.id"))
    )
    team_id: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.teams.id"))
    )
    role: TeamRole = Field(
        default=TeamRole.MEMBER,
        sa_column=Column(SQLEnum(TeamRole, schema="app_schema")),
    )
    joined_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )

    # Relationships
    user: User = Relationship(back_populates="team_memberships")
    team: Team = Relationship(back_populates="members")


class ApiToken(SQLModel, table=True):
    __tablename__ = "api_tokens"
    __table_args__ = {"schema": "app_schema"}

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    user_id: uuid.UUID = Field(
        sa_column=Column(
            UUID, ForeignKey("app_schema.users.id", ondelete="CASCADE"), index=True
        )
    )
    name: str = Field(max_length=255)  # User-defined token name
    token_hash: bytes = Field(
        sa_column=Column(
            BYTEA(_ApiKeyAuthorizerSettings.TOKEN_HASH_LENGTH),
            unique=True,
            index=True,
            nullable=False,
        )
    )  # bcrypt hash (62 bytes fixed)
    token_prefix: str = Field(
        max_length=_ApiKeyAuthorizerSettings.PREVIEW_LENGTH
    )  # "dsp_a1b2c3d4" for display
    scopes: t.List[TokenScope] = Field(
        default_factory=lambda: [TokenScope.READ],
        sa_column=Column(ARRAY(SQLEnum(TokenScope, schema="app_schema"))),
    )
    is_active: bool = Field(default=True, index=True)
    last_used_at: t.Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True))
    )
    expires_at: t.Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), index=True)
    )  # Optional expiration
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )

    # Relationships
    user: User = Relationship(back_populates="api_tokens")
