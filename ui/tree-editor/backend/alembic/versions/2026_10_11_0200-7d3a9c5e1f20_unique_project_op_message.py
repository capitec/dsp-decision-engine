"""unique_project_op_message

Revision ID: 7d3a9c5e1f20
Revises: 4f2b8c1e9a37
Create Date: 2026-10-11 02:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
from dsp_decikit.settings import settings

# revision identifiers, used by Alembic.
revision: str = "7d3a9c5e1f20"
down_revision: Union[str, Sequence[str], None] = "4f2b8c1e9a37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: a client's message is logged at most once."""
    op.create_index("uq_project_ops_message", "project_ops", ["project_id", "client_id", "msg_id"], unique=True,
                    schema=settings.db.pg_schema)


def downgrade() -> None:
    """Downgrade schema: drop the one-row-per-message index."""
    op.drop_index("uq_project_ops_message", table_name="project_ops", schema=settings.db.pg_schema)
