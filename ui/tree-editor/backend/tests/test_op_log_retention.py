"""The op log keeps a bounded number of versions, against a real Postgres (TREE_EDITOR_DB=1, as `test_live_sync.py`):
every few versions it stores the whole document and trims what is older than the count or age it keeps, and a
client or an offline batch that needs a trimmed version gets a snapshot or a clear refusal.
"""

import pytest

from dsp_decikit.services import live_sync
from dsp_decikit.services.live_sync import Rejected
from dsp_decikit.settings import settings
from tests.test_live_offline import merge, offline  # noqa: F401
from tests.test_live_sync import Socket, db, log, name, project  # noqa: F401


@pytest.fixture
def small_log(offline, monkeypatch):
    """A project logged from version 1 whose log snapshots every 3 versions and keeps the last 2 (plus the rows back
    to a snapshot)."""
    monkeypatch.setattr(live_sync, "SNAPSHOT_EVERY", 3)
    monkeypatch.setattr(settings, "live_log_keep_versions", 2)
    return offline


async def write_until(project_id, stored, last) -> dict:
    """Edit rule a once per version up to `last`; returns the stored document of each version."""
    from dsp_decikit.services.live_sync import apply_message
    docs = {}
    version = (await stored()).version
    while version < last:
        version = (await apply_message(project_id, "c1", f"m{version + 1}", version, [name("a", f"v{version + 1}")])).version
        docs[version] = (await stored()).project_data
    return docs


async def snapshots(project_id):
    from sqlalchemy import select
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import ProjectOp

    async with db_session() as session:
        rows = await session.execute(select(ProjectOp.version, ProjectOp.snapshot)
                                     .where(ProjectOp.project_id == project_id).order_by(ProjectOp.version))
        return {v: s for v, s in rows}


async def document_at(project_id, version):
    from dsp_decikit.db.session import db_session

    async with db_session() as session:
        return await live_sync._document_at(session, project_id, version)


@db
@pytest.mark.asyncio
async def test_every_nth_version_stores_the_whole_document_as_a_snapshot(offline, monkeypatch):
    monkeypatch.setattr(live_sync, "SNAPSHOT_EVERY", 3)
    project_id, stored = offline

    docs = await write_until(project_id, stored, 6)

    kept = await snapshots(project_id)
    assert {v for v, s in kept.items() if s is not None} == {3, 6}
    assert kept[3] == docs[3] and kept[6] == docs[6]


@db
@pytest.mark.asyncio
async def test_writing_a_snapshot_trims_versions_beyond_the_count_down_to_a_snapshot(small_log):
    project_id, stored = small_log

    await write_until(project_id, stored, 9)

    # The oldest kept row is a snapshot, so every kept version can still be rebuilt.
    assert [v for v, *_ in await log(project_id)] == [6, 7, 8, 9]


@db
@pytest.mark.asyncio
async def test_versions_older_than_the_age_limit_are_trimmed_down_to_a_snapshot(offline, monkeypatch):
    from datetime import timedelta
    from sqlalchemy import func, update
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import ProjectOp
    monkeypatch.setattr(live_sync, "SNAPSHOT_EVERY", 3)
    monkeypatch.setattr(settings, "live_log_keep_days", 1)
    project_id, stored = offline
    await write_until(project_id, stored, 5)
    async with db_session() as session:
        await session.execute(update(ProjectOp).where(ProjectOp.project_id == project_id, ProjectOp.version <= 4)
                              .values(created_at=func.now() - timedelta(days=2)))
        await session.commit()

    await write_until(project_id, stored, 6)

    assert [v for v, *_ in await log(project_id)] == [3, 4, 5, 6]


@db
@pytest.mark.asyncio
async def test_every_kept_version_is_rebuilt_from_the_log_after_trimming(small_log):
    project_id, stored = small_log

    docs = await write_until(project_id, stored, 11)

    assert [v for v, *_ in await log(project_id)] == [6, 7, 8, 9, 10, 11]
    for version in range(6, 12):
        assert await document_at(project_id, version) == docs[version]


@db
@pytest.mark.asyncio
async def test_a_version_is_rebuilt_from_the_newest_snapshot_at_or_before_it(offline, monkeypatch):
    from sqlalchemy import delete
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import ProjectOp
    monkeypatch.setattr(live_sync, "SNAPSHOT_EVERY", 3)
    project_id, stored = offline
    docs = await write_until(project_id, stored, 7)
    # Without the versions between the first row and the snapshot, a replay from the first row would be wrong.
    async with db_session() as session:
        await session.execute(delete(ProjectOp).where(ProjectOp.project_id == project_id,
                                                      ProjectOp.version.between(2, 5)))
        await session.commit()

    assert await document_at(project_id, 7) == docs[7]


@db
@pytest.mark.asyncio
async def test_a_version_after_a_hole_in_the_log_is_not_rebuilt(offline, monkeypatch):
    # A hole means a write that was not logged; replaying across it would build the wrong document.
    from sqlalchemy import delete
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import ProjectOp
    project_id, stored = offline
    await write_until(project_id, stored, 4)
    async with db_session() as session:
        await session.execute(delete(ProjectOp).where(ProjectOp.project_id == project_id, ProjectOp.version == 3))
        await session.commit()

    with pytest.raises(Rejected):
        await document_at(project_id, 4)


# ---- a base older than the log keeps ---------------------------------------------------------------------------

@db
@pytest.mark.asyncio
async def test_a_client_reconnecting_from_a_trimmed_version_gets_a_snapshot(small_log):
    from dsp_decikit.services.live_hub import Client, Hub
    project_id, stored = small_log
    await write_until(project_id, stored, 9)
    client = Client(Socket(), "c2")

    await Hub().join(project_id, client, 2)

    first, last = client.ws.sent[0], client.ws.sent[-1]
    assert (first["type"], first["version"], first["doc"]) == ("snapshot", 9, (await stored()).project_data)
    assert (last["type"], last["version"]) == ("ready", 9)


@db
@pytest.mark.asyncio
async def test_a_client_is_not_caught_up_across_a_version_trimmed_while_it_joined(offline):
    # The join read the project at version 3; meanwhile a write logged version 4 and trimmed version 2 away.
    from sqlalchemy import delete, update
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import Project, ProjectOp
    from dsp_decikit.services.live_hub import Client, Hub
    project_id, stored = offline
    await write_until(project_id, stored, 4)
    async with db_session() as session:
        await session.execute(delete(ProjectOp).where(ProjectOp.project_id == project_id, ProjectOp.version == 2))
        await session.execute(update(Project).where(Project.id == project_id).values(version=3))
        await session.commit()
    client = Client(Socket(), "c2")

    await Hub().join(project_id, client, 1)

    assert client.ws.sent[0]["type"] == "snapshot"


@db
@pytest.mark.asyncio
@pytest.mark.parametrize("base", [2, 5], ids=["its versions are gone", "only its own document is gone"])
async def test_an_offline_batch_on_a_trimmed_base_is_refused_naming_the_version(small_log, base):
    project_id, stored = small_log
    await write_until(project_id, stored, 9)

    with pytest.raises(Rejected) as e:
        await merge(project_id, [name("b", "Offline")], base=base)

    assert f"version {base}" in str(e.value) and e.value.conflicts == []
    assert (await stored()).version == 9


@db
@pytest.mark.asyncio
async def test_an_online_message_on_a_trimmed_base_is_refused_not_silently_lost(small_log):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = small_log
    await write_until(project_id, stored, 9)

    with pytest.raises(Rejected):
        await apply_message(project_id, "c2", "late", 2, [name("b", "Online")])

    assert (await stored()).version == 9


@db
@pytest.mark.asyncio
async def test_concurrent_writes_and_offline_merges_while_trimming_keep_the_log_rebuilding_the_stored_document(
        small_log, monkeypatch):
    import asyncio
    from dsp_decikit.services.live_sync import apply_message
    monkeypatch.setattr(settings, "live_log_keep_versions", 6)
    project_id, stored = small_log
    await write_until(project_id, stored, 4)

    async def write(i):
        base = (await stored()).version - i % 3
        try:
            await apply_message(project_id, f"c{i % 4}", f"w{i}", base, [name("abc"[i % 3], f"v{i}")],
                                offline=i % 4 == 0)
        except Rejected:
            pass  # a conflict, or a base trimmed meanwhile: refused whole, nothing stored

    await asyncio.gather(*(write(i) for i in range(60)))

    head = await stored()
    versions = [v for v, *_ in await log(project_id)]
    assert versions == list(range(versions[0], head.version + 1)) and head.version > 10
    for version in versions[:-1]:
        await document_at(project_id, version)
    assert await document_at(project_id, head.version) == head.project_data
