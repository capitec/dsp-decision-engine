"""project_op_snapshots

Revision ID: 633e3c45d4e8
Revises: 8e4b2d6f0a91
Create Date: 2026-10-11 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from dsp_decikit.settings import settings

# revision identifiers, used by Alembic.
revision: str = "633e3c45d4e8"
down_revision: Union[str, Sequence[str], None] = "8e4b2d6f0a91"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: a log row can hold the whole document of its version."""
    op.add_column("project_ops", sa.Column("snapshot", postgresql.JSONB(), nullable=True),
                  schema=settings.db.pg_schema)


def downgrade() -> None:
    """Downgrade schema: drop the snapshots."""
    op.drop_column("project_ops", "snapshot", schema=settings.db.pg_schema)
