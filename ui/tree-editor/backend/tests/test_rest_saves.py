"""A REST document save in the op log, against a real Postgres (TREE_EDITOR_DB=1, as `test_live_sync.py`): it is
logged as the diff from the stored document, so the log still rebuilds every kept version, live clients apply it,
and only edits that overlap what it changed conflict with it.
"""

import asyncio
import copy

import pytest

from dsp_decikit.services.live_ops import apply
from dsp_decikit.services.live_sync import Rejected
from tests.test_live_offline import merge, names, offline  # noqa: F401
from tests.test_live_sync import Socket, db, log, name, project  # noqa: F401
from tests.test_op_log_retention import document_at, small_log  # noqa: F401


async def rest_save(project_id, edit) -> dict:
    """Save the stored document with `edit` (a function changing it in place) as the REST API does; returns it."""
    from dsp_decikit.db.session import db_session
    from dsp_decikit.services.live_sync import _locked, save_version

    async with db_session() as session:
        current = await _locked(session, project_id, lock=False)
        doc, base = copy.deepcopy(current.project_data), current.version
    edit(doc)
    async with db_session() as session:
        await save_version(session, project_id, base, {"project_data": doc}, "rest", f"save-{base}")
        await session.commit()
    return doc


def rename(index, value):
    def edit(doc):
        doc["rules"][index]["meta"]["name"] = value
    return edit


@db
@pytest.mark.asyncio
async def test_a_rest_save_is_logged_as_the_diff_from_the_stored_document(offline):
    project_id, stored = offline

    doc = await rest_save(project_id, rename(1, "Saved"))

    assert (await stored()).project_data == doc
    assert (await log(project_id))[-1] == (2, "rest", "save-1", [name("b", "Saved")])


@db
@pytest.mark.asyncio
async def test_a_rest_save_that_changes_nothing_in_the_document_still_makes_a_version_logging_no_ops(offline):
    project_id, stored = offline

    await rest_save(project_id, lambda doc: None)

    assert (await stored()).version == 2
    assert (await log(project_id))[-1] == (2, "rest", "save-1", [])


@db
@pytest.mark.asyncio
async def test_a_new_projects_first_version_is_whole_so_rest_saves_after_it_are_rebuilt(project):
    from dsp_decikit.db.session import db_session
    from dsp_decikit.services.live_sync import first_version
    project_id, stored = project
    async with db_session() as session:
        session.add(first_version(await stored()))
        await session.commit()
    docs = {1: (await stored()).project_data}

    for version in (2, 3):
        docs[version] = await rest_save(project_id, rename(0, f"v{version}"))

    assert (await log(project_id))[0][3] == [{"op": "replace", "path": "", "value": docs[1]}]
    for version, doc in docs.items():
        assert await document_at(project_id, version) == doc


@db
@pytest.mark.asyncio
async def test_trimming_a_log_of_rest_saves_keeps_back_to_a_whole_row_so_every_kept_version_is_rebuilt(small_log):
    project_id, stored = small_log
    docs = {}

    for version in range(2, 12):
        docs[version] = await rest_save(project_id, rename(version % 3, f"v{version}"))

    # Snapshots every 3 versions, 2 kept: the oldest kept row is the snapshot at 6, not a REST save after it.
    assert [v for v, *_ in await log(project_id)] == [6, 7, 8, 9, 10, 11]
    for version in range(6, 12):
        assert await document_at(project_id, version) == docs[version]


@db
@pytest.mark.asyncio
async def test_a_live_client_applying_a_rest_save_from_the_fan_out_gets_the_saved_document(offline):
    from dsp_decikit.services.live_hub import Client, Hub
    project_id, stored = offline
    seen = (await stored()).project_data
    hub, client = Hub(), Client(Socket(), "live")
    await hub.join(project_id, client, 1)

    def edit(doc):
        doc["rules"] = doc["rules"][::-1]
        doc["rules"][0]["meta"]["name"] = "Saved"
        del doc["rules"][1]
        doc["parameters"]["limit"]["default_value"] = 7
    saved = await rest_save(project_id, edit)
    hub.pump_soon(project_id)
    await asyncio.gather(*[r.pumping for r in hub.rooms.values() if r.pumping])

    [version] = [m for m in client.ws.sent if m["type"] == "applied"]
    assert version["version"] == 2 and apply(seen, version["ops"]) == (saved, [])


@db
@pytest.mark.asyncio
async def test_an_offline_batch_that_overlaps_nothing_a_rest_save_changed_merges(offline):
    project_id, stored = offline
    await rest_save(project_id, rename(0, "Saved"))

    applied = await merge(project_id, [name("b", "Offline")])

    assert (applied.version, names(await stored())) == (3, ["Saved", "Offline", "C"])


@db
@pytest.mark.asyncio
async def test_an_offline_batch_overlapping_a_rest_save_is_refused_naming_the_path(offline):
    project_id, stored = offline
    await rest_save(project_id, rename(0, "Saved"))

    with pytest.raises(Rejected) as e:
        await merge(project_id, [name("a", "Offline"), name("b", "Offline")])

    assert e.value.conflicts == ["/rules/a/meta/name"]
    assert (await stored()).version == 2


@db
@pytest.mark.asyncio
async def test_an_online_edit_made_before_a_rest_save_is_dropped_only_inside_what_the_save_changed(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await rest_save(project_id, lambda doc: doc["rules"].pop(0))

    applied = await apply_message(project_id, "live", "m1", 1, [name("a", "Online"), name("b", "Online")])

    assert applied.dropped == [name("a", "Online")]
    assert names(await stored()) == ["Online", "C"]


@db
@pytest.mark.asyncio
async def test_a_rest_save_too_deep_to_diff_is_logged_whole(offline):
    project_id, stored = offline

    def nest(doc):
        deep = doc["rules"][0]["meta"]
        for _ in range(200):
            deep["x"] = {}
            deep = deep["x"]

    doc = await rest_save(project_id, nest)

    assert (await stored()).project_data == doc
    assert (await log(project_id))[-1] == (2, "rest", "save-1", [{"op": "replace", "path": "", "value": doc}])
