"""project_op_parts

Revision ID: 8e4b2d6f0a91
Revises: 7d3a9c5e1f20
Create Date: 2026-10-11 03:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from dsp_decikit.settings import settings

# revision identifiers, used by Alembic.
revision: str = "8e4b2d6f0a91"
down_revision: Union[str, Sequence[str], None] = "7d3a9c5e1f20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: a logged offline batch names the messages it contains."""
    op.add_column("project_ops", sa.Column("parts", postgresql.JSONB(), nullable=False, server_default="[]"),
                  schema=settings.db.pg_schema)


def downgrade() -> None:
    """Downgrade schema: drop the contained-message ids."""
    op.drop_column("project_ops", "parts", schema=settings.db.pg_schema)
