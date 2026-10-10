"""add_project_version

Revision ID: 9c4e1d7a52b8
Revises: 31bb7952b593
Create Date: 2026-10-10 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from dsp_decikit.settings import settings

# revision identifiers, used by Alembic.
revision: str = "9c4e1d7a52b8"
down_revision: Union[str, Sequence[str], None] = "31bb7952b593"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: a version that every save of a project's document bumps."""
    op.add_column(
        "projects",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        schema=settings.db.pg_schema,
    )


def downgrade() -> None:
    """Downgrade schema: drop the project version."""
    op.drop_column("projects", "version", schema=settings.db.pg_schema)
