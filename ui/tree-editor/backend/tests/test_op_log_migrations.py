"""The op-log migrations on a database that ran an earlier step, against a real Postgres (TREE_EDITOR_DB=1, as
`test_live_sync.py`): each test migrates a scratch database of its own to the step-1 schema, stores what step 0 or
step 1 left behind, and upgrades to head.
"""

import json
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio

from dsp_decikit.settings import settings
from tests.test_live_sync import RULE_A, RULE_B, db, flat, name

pytestmark = db
BACKEND = Path(__file__).resolve().parents[1]
STEP_1 = "4f2b8c1e9a37"  # the project_ops table, before the one-row-per-message index


def alembic(database, *args):
    done = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=BACKEND, capture_output=True, text=True,
                          env={**os.environ, "DSP_DB__DATABASE": database})
    assert done.returncode == 0, done.stdout + done.stderr


@pytest_asyncio.fixture
async def step_1():
    """A scratch database migrated to the step-1 schema; yields its engine and name."""
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    database = f"oplog_{uuid.uuid4().hex[:12]}"
    server = create_async_engine(settings.db.database_url, isolation_level="AUTOCOMMIT")
    async with server.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{database}"'))
    alembic(database, "upgrade", STEP_1)
    engine = create_async_engine(make_url(settings.db.database_url).set(database=database),
                                 execution_options=settings.db.get_execution_options())
    yield engine, database
    await engine.dispose()
    async with server.connect() as conn:
        await conn.execute(text(f'DROP DATABASE "{database}" WITH (FORCE)'))
    await server.dispose()


async def new_project(engine, doc, version) -> uuid.UUID:
    from sqlalchemy.ext.asyncio import AsyncSession
    from dsp_decikit.db.models import Project, ProjectType, User

    async with AsyncSession(engine) as session:
        user = User(email=f"{uuid.uuid4().hex[:10]}@example.com")
        session.add(user)
        await session.flush()
        project = Project(name="old", project_type=ProjectType.FLAT_RULE, project_data=doc, created_by=user.id,
                          version=version)
        session.add(project)
        await session.commit()
        return project.id


async def log_row(engine, project_id, version, client_id, msg_id, ops):
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text(f"INSERT INTO {settings.db.pg_schema}.project_ops (project_id, version, client_id, "
                                "msg_id, ops) VALUES (:p, :v, :c, :m, CAST(:ops AS jsonb))"),
                           {"p": project_id, "v": version, "c": client_id, "m": msg_id, "ops": json.dumps(ops)})


async def logged(engine, project_id):
    from sqlalchemy import text

    async with engine.connect() as conn:
        rows = await conn.execute(text(f"SELECT version, client_id, msg_id FROM {settings.db.pg_schema}.project_ops "
                                       "WHERE project_id = :p ORDER BY version"), {"p": project_id})
        return [tuple(r) for r in rows]


@pytest.fixture
def on(monkeypatch):
    """Point the live-sync writes at another database's engine."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from dsp_decikit.services import live_sync

    def use(engine):
        @asynccontextmanager
        async def session():
            async with AsyncSession(engine, expire_on_commit=False) as s:
                yield s
        monkeypatch.setattr(live_sync, "db_session", session)
    return use


async def stored(engine, project_id):
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from dsp_decikit.db.models import Project

    async with AsyncSession(engine) as session:
        return (await session.execute(select(Project).where(Project.id == project_id))).scalar_one()


@pytest.mark.asyncio
async def test_a_project_from_before_the_log_accepts_an_offline_batch_after_the_migration(step_1, on):
    from dsp_decikit.services.live_sync import apply_message
    engine, database = step_1
    project_id = await new_project(engine, flat(RULE_A, RULE_B), version=3)  # saved twice in step 0, no log

    alembic(database, "upgrade", "head")
    on(engine)
    await apply_message(project_id, "online", "m1", 3, [name("a", "Online")])
    applied = await apply_message(project_id, "offline", "batch", 3, [name("b", "Offline")], offline=True)

    row = await stored(engine, project_id)
    assert applied.version == row.version == 5
    assert [r["meta"]["name"] for r in row.project_data["rules"]] == ["Online", "Offline"]


@pytest.mark.asyncio
async def test_a_project_with_only_step_one_op_rows_accepts_an_offline_batch_after_the_migration(step_1, on):
    from dsp_decikit.services.live_sync import apply_message
    engine, database = step_1
    renamed = flat({**RULE_A, "meta": {"name": "One"}}, {**RULE_B, "meta": {"name": "Two"}})
    project_id = await new_project(engine, renamed, version=3)  # created in step 1: no row for version 1
    await log_row(engine, project_id, 2, "c1", "m1", [name("a", "One")])
    await log_row(engine, project_id, 3, "c1", "m2", [name("b", "Two")])

    alembic(database, "upgrade", "head")
    on(engine)
    await apply_message(project_id, "online", "m3", 3, [name("a", "Online")])
    applied = await apply_message(project_id, "offline", "batch", 3, [name("b", "Offline")], offline=True)

    row = await stored(engine, project_id)
    assert applied.version == row.version == 5
    assert [r["meta"]["name"] for r in row.project_data["rules"]] == ["Online", "Offline"]
    assert [v for v, *_ in await logged(engine, project_id)] == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_the_one_row_per_message_migration_keeps_every_version_of_a_message_logged_twice(step_1):
    # A resend applied twice in step 1: the first row keeps the message id, later ones are renamed.
    engine, database = step_1
    project_id = await new_project(engine, flat(RULE_A), version=5)
    for version, client_id, msg_id in [(2, "c1", "m1"), (3, "c1", "m1"), (4, "c2", "m1"), (5, "c1", "m1")]:
        await log_row(engine, project_id, version, client_id, msg_id, [name("a", f"v{version}")])

    alembic(database, "upgrade", "head")

    rows = await logged(engine, project_id)
    assert [v for v, *_ in rows] == [2, 3, 4, 5]
    assert rows[0] == (2, "c1", "m1") and rows[2] == (4, "c2", "m1")
    assert len({(c, m) for _, c, m in rows}) == 4


@pytest.mark.asyncio
async def test_the_op_log_migrations_go_down_to_step_1_and_up_again(step_1):
    engine, database = step_1
    project_id = await new_project(engine, flat(RULE_A), version=3)
    await log_row(engine, project_id, 2, "c1", "m1", [name("a", "x")])
    await log_row(engine, project_id, 3, "c1", "m1", [name("a", "y")])

    alembic(database, "upgrade", "head")
    alembic(database, "downgrade", STEP_1)
    alembic(database, "upgrade", "head")

    assert [v for v, *_ in await logged(engine, project_id)] == [2, 3]
