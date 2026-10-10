"""consolidate_projects_and_flat_rules

Revision ID: a6ec477263a4
Revises: f80757a4da1f
Create Date: 2026-05-06 14:20:50.230612

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel.sql.sqltypes
from dsp_decikit.settings import settings

# revision identifiers, used by Alembic.
revision: str = "a6ec477263a4"
down_revision: Union[str, Sequence[str], None] = "f80757a4da1f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Consolidate projects and flat_rules tables with data migration."""
    schema_name = settings.db.pg_schema

    # Step 1: Create project_type enum
    op.execute(f"CREATE TYPE {schema_name}.projecttype AS ENUM ('flat_rule', 'tree');")

    # Step 2: Add new columns to projects table
    op.add_column(
        "projects",
        sa.Column(
            "project_type",
            sa.Enum("flat_rule", "tree", name="projecttype", schema=schema_name),
            nullable=True,  # Temporarily nullable for migration
        ),
        schema=schema_name,
    )
    op.add_column(
        "projects",
        sa.Column(
            "project_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        schema=schema_name,
    )

    # Step 3: Migrate data from project_revisions to projects
    # Get the latest revision for each project
    op.execute(f"""
        UPDATE {schema_name}.projects p
        SET
            project_data = pr.full_state,
            project_type = 'tree'
        FROM (
            SELECT DISTINCT ON (project_id)
                project_id,
                full_state
            FROM {schema_name}.project_revisions
            ORDER BY project_id, revision_number DESC
        ) pr
        WHERE p.id = pr.project_id;
    """)

    # Step 4: Set default project_type for any projects without revisions
    op.execute(f"""
        UPDATE {schema_name}.projects
        SET project_type = 'tree', project_data = '{{}}'::jsonb
        WHERE project_type IS NULL;
    """)

    # Step 5: Migrate data from flat_rule_projects to projects
    op.execute(f"""
        INSERT INTO {schema_name}.projects (
            id, name, description, project_type, created_by, project_data, created_at, updated_at
        )
        SELECT
            id, name, description, 'flat_rule', created_by, project_data, created_at, updated_at
        FROM {schema_name}.flat_rule_projects;
    """)

    # Step 6: Make project_type and project_data NOT NULL now that data is migrated
    op.alter_column(
        "projects",
        "project_type",
        nullable=False,
        schema=schema_name,
    )
    op.alter_column(
        "projects",
        "project_data",
        nullable=False,
        schema=schema_name,
    )

    # Step 7: Drop old tables
    # First drop the foreign key constraints and relationships
    op.drop_constraint(
        "project_revisions_project_id_fkey",
        "project_revisions",
        type_="foreignkey",
        schema=schema_name,
    )
    # Not dropping these just yet as they contain old data and might be useful for rollback
    # op.drop_table("project_revisions", schema=schema_name)
    # op.drop_table("flat_rule_projects", schema=schema_name)
    # op.execute(f"DROP TYPE IF EXISTS {schema_name}.revisiontype;")


def downgrade() -> None:
    """Downgrade schema - recreate separate tables."""
    schema_name = settings.db.pg_schema

    # We didnt drop these just yet
    # op.execute(f"CREATE TYPE {schema_name}.revisiontype AS ENUM ('manual', 'auto_save', 'deployment');")
    # op.create_table(
    #     "project_revisions",
    #     sa.Column("id", sa.UUID(), nullable=False),
    #     sa.Column("project_id", sa.UUID(), nullable=False),
    #     sa.Column("revision_number", sa.Integer(), nullable=False),
    #     sa.Column("full_state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    #     sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    #     sa.Column("created_by", sa.UUID(), nullable=False),
    #     sa.Column(
    #         "revision_type",
    #         sa.Enum("manual", "auto_save", "deployment", name="revisiontype", schema=schema_name),
    #         nullable=False,
    #     ),
    #     sa.Column("is_deployment_ready", sa.Boolean(), nullable=False),
    #     sa.Column("contributing_users", postgresql.ARRAY(sa.UUID()), nullable=True),
    #     sa.ForeignKeyConstraint(["created_by"], [f"{schema_name}.users.id"]),
    #     sa.ForeignKeyConstraint(["project_id"], [f"{schema_name}.projects.id"]),
    #     sa.PrimaryKeyConstraint("id"),
    #     sa.UniqueConstraint("project_id", "revision_number"),
    #     schema=schema_name,
    # )

    # # Recreate flat_rule_projects table
    # op.create_table(
    #     "flat_rule_projects",
    #     sa.Column("id", sa.UUID(), nullable=False),
    #     sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    #     sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    #     sa.Column("created_by", sa.UUID(), nullable=True),
    #     sa.Column("project_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    #     sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    #     sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    #     sa.ForeignKeyConstraint(["created_by"], [f"{schema_name}.users.id"]),
    #     sa.PrimaryKeyConstraint("id"),
    #     schema=schema_name,
    # )

    # Migrate data back from projects to project_revisions
    # op.execute(f"""
    #     INSERT INTO {schema_name}.project_revisions (
    #         id, project_id, revision_number, full_state, created_at, created_by,
    #         revision_type, is_deployment_ready, contributing_users
    #     )
    #     SELECT
    #         gen_random_uuid(), id, 1, project_data, created_at, created_by,
    #         'manual', false, ARRAY[]::uuid[]
    #     FROM {schema_name}.projects
    #     WHERE project_type = 'tree';
    # """)

    # # Migrate data back from projects to flat_rule_projects
    # op.execute(f"""
    #     INSERT INTO {schema_name}.flat_rule_projects (
    #         id, name, description, created_by, project_data, created_at, updated_at
    #     )
    #     SELECT
    #         id, name, description, created_by, project_data, created_at, updated_at
    #     FROM {schema_name}.projects
    #     WHERE project_type = 'flat_rule';
    # """)

    # Drop new columns from projects
    op.drop_column("projects", "project_data", schema=schema_name)
    op.drop_column("projects", "project_type", schema=schema_name)

    # Drop projecttype enum
    op.execute(f"DROP TYPE IF EXISTS {schema_name}.projecttype;")
