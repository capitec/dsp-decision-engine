"""add flat rules

Revision ID: f80757a4da1f
Revises: 65bd9f89973f
Create Date: 2026-03-18 08:58:03.765336

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel.sql.sqltypes
from dsp_decider.settings import settings

# revision identifiers, used by Alembic.
revision: str = "f80757a4da1f"
down_revision: Union[str, Sequence[str], None] = "65bd9f89973f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add flat rule projects table."""
    schema_name = settings.db.pg_schema

    # Create flat_rule_projects table with complete PED-compatible JSONB structure
    op.create_table(
        "flat_rule_projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column(
            "project_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["created_by"], [f"{schema_name}.users.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=schema_name,
    )


def downgrade() -> None:
    """Downgrade schema to remove flat rule projects table."""
    schema_name = settings.db.pg_schema
    # Drop the single flat rule projects table
    op.drop_table("flat_rule_projects", schema=schema_name)
