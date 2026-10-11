"""add_project_ops

Revision ID: 4f2b8c1e9a37
Revises: 9c4e1d7a52b8
Create Date: 2026-10-10 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from dsp_decikit.settings import settings

# revision identifiers, used by Alembic.
revision: str = "4f2b8c1e9a37"
down_revision: Union[str, Sequence[str], None] = "9c4e1d7a52b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: the log of every project version, the ops that made it (kept forever for now)."""
    schema_name = settings.db.pg_schema
    op.create_table(
        "project_ops",
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("msg_id", sa.Text(), nullable=False),
        sa.Column("ops", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["project_id"], [f"{schema_name}.projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "version"),
        schema=schema_name,
    )


def downgrade() -> None:
    """Downgrade schema: drop the project ops log."""
    op.drop_table("project_ops", schema=settings.db.pg_schema)
