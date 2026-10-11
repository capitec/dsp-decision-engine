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
    """Upgrade schema: a log row can hold the whole document of its version, and every project's current version
    is rebuildable from the log (projects saved before the log, or only live-edited, had no whole document in it)."""
    schema = settings.db.pg_schema
    op.add_column("project_ops", sa.Column("snapshot", postgresql.JSONB(), nullable=True), schema=schema)
    whole = f"""SELECT 1 FROM {schema}.project_ops w WHERE w.project_id = p.id
                AND (w.snapshot IS NOT NULL OR w.ops->0->>'path' = '')"""
    op.execute(f"""
        UPDATE {schema}.project_ops o SET snapshot = p.project_data FROM {schema}.projects p
        WHERE o.project_id = p.id AND o.version = p.version AND p.project_data IS NOT NULL AND NOT EXISTS ({whole})
    """)
    op.execute(f"""
        INSERT INTO {schema}.project_ops (project_id, version, client_id, msg_id, ops)
        SELECT p.id, p.version, 'server', 'backfilled',
               jsonb_build_array(jsonb_build_object('op', 'replace', 'path', '', 'value', p.project_data))
        FROM {schema}.projects p WHERE p.project_data IS NOT NULL AND NOT EXISTS ({whole})
    """)


def downgrade() -> None:
    """Downgrade schema: drop the snapshots (the backfilled rows stay: they are ordinary whole-document versions)."""
    op.drop_column("project_ops", "snapshot", schema=settings.db.pg_schema)
