"""add_folder_structure

Revision ID: 31bb7952b593
Revises: a6ec477263a4
Create Date: 2026-05-12 15:39:55.432877

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from dsp_decider.settings import settings

# revision identifiers, used by Alembic.
revision: str = "31bb7952b593"
down_revision: Union[str, Sequence[str], None] = "a6ec477263a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add folder structure."""
    schema_name = settings.db.pg_schema

    # Create folders table
    op.create_table(
        "folders",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parent_id", postgresql.UUID(), nullable=True),
        sa.Column("created_by", postgresql.UUID(), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], [f"{schema_name}.users.id"]),
        sa.ForeignKeyConstraint(["parent_id"], [f"{schema_name}.folders.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "name", "parent_id", "created_by", name="uq_folder_name_parent_creator"
        ),
        schema=schema_name,
    )

    # Create indexes on folders table
    op.create_index("ix_folders_name", "folders", ["name"], schema=schema_name)
    op.create_index(
        "ix_folders_parent_id", "folders", ["parent_id"], schema=schema_name
    )
    op.create_index(
        "ix_folders_created_by", "folders", ["created_by"], schema=schema_name
    )

    # Get existing enum types (they already exist from project_shares table)
    sharetype_enum = postgresql.ENUM(
        "USER", "TEAM", name="sharetype", schema=schema_name, create_type=False
    )
    accessmode_enum = postgresql.ENUM(
        "READ",
        "WRITE",
        "ADMIN",
        name="accessmode",
        schema=schema_name,
        create_type=False,
    )

    # Create folder_shares table
    op.create_table(
        "folder_shares",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("folder_id", postgresql.UUID(), nullable=False),
        sa.Column("share_type", sharetype_enum, nullable=False),
        sa.Column("user_id", postgresql.UUID(), nullable=True),
        sa.Column("team_id", postgresql.UUID(), nullable=True),
        sa.Column("access_mode", accessmode_enum, nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("created_by", postgresql.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["folder_id"], [f"{schema_name}.folders.id"]),
        sa.ForeignKeyConstraint(["user_id"], [f"{schema_name}.users.id"]),
        sa.ForeignKeyConstraint(["team_id"], [f"{schema_name}.teams.id"]),
        sa.ForeignKeyConstraint(["created_by"], [f"{schema_name}.users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "folder_id",
            "share_type",
            "user_id",
            "team_id",
            name="uq_folder_share_unique",
        ),
        schema=schema_name,
    )

    # Create indexes on folder_shares table
    op.create_index(
        "ix_folder_shares_folder_id", "folder_shares", ["folder_id"], schema=schema_name
    )
    op.create_index(
        "ix_folder_shares_user_id", "folder_shares", ["user_id"], schema=schema_name
    )
    op.create_index(
        "ix_folder_shares_team_id", "folder_shares", ["team_id"], schema=schema_name
    )

    # Add folder_id column to projects table
    op.add_column(
        "projects",
        sa.Column("folder_id", postgresql.UUID(), nullable=True),
        schema=schema_name,
    )
    op.create_foreign_key(
        "fk_projects_folder_id",
        "projects",
        "folders",
        ["folder_id"],
        ["id"],
        source_schema=schema_name,
        referent_schema=schema_name,
    )
    op.create_index(
        "ix_projects_folder_id", "projects", ["folder_id"], schema=schema_name
    )


def downgrade() -> None:
    """Downgrade schema to remove folder structure."""
    schema_name = settings.db.pg_schema

    # Remove folder_id from projects table
    op.drop_index("ix_projects_folder_id", table_name="projects", schema=schema_name)
    op.drop_constraint(
        "fk_projects_folder_id", "projects", type_="foreignkey", schema=schema_name
    )
    op.drop_column("projects", "folder_id", schema=schema_name)

    # Drop folder_shares table
    op.drop_index(
        "ix_folder_shares_team_id", table_name="folder_shares", schema=schema_name
    )
    op.drop_index(
        "ix_folder_shares_user_id", table_name="folder_shares", schema=schema_name
    )
    op.drop_index(
        "ix_folder_shares_folder_id", table_name="folder_shares", schema=schema_name
    )
    op.drop_table("folder_shares", schema=schema_name)

    # Drop folders table
    op.drop_index("ix_folders_created_by", table_name="folders", schema=schema_name)
    op.drop_index("ix_folders_parent_id", table_name="folders", schema=schema_name)
    op.drop_index("ix_folders_name", table_name="folders", schema=schema_name)
    op.drop_table("folders", schema=schema_name)
