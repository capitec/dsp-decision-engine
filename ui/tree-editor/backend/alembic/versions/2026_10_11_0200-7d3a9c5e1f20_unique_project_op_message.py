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
    schema = settings.db.pg_schema
    # A resend applied twice before this index keeps both versions (the log must stay contiguous); the later rows
    # get their version appended to the message id.
    op.execute(f"""
        UPDATE {schema}.project_ops o SET msg_id = o.msg_id || '#' || o.version
        WHERE EXISTS (SELECT 1 FROM {schema}.project_ops e WHERE e.project_id = o.project_id
                      AND e.client_id = o.client_id AND e.msg_id = o.msg_id AND e.version < o.version)
    """)
    op.create_index("uq_project_ops_message", "project_ops", ["project_id", "client_id", "msg_id"], unique=True,
                    schema=settings.db.pg_schema)


def downgrade() -> None:
    """Downgrade schema: drop the one-row-per-message index."""
    op.drop_index("uq_project_ops_message", table_name="project_ops", schema=settings.db.pg_schema)
