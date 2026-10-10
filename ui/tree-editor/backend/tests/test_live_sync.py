"""The server side of live editing: message checks, id stamping, and the online rule against a real Postgres.

The database tests need a migrated Postgres (the e2e stack's) and run only when LIVE_SYNC_DB=1, with
DSP_DB__PORT pointing at it: the row lock and the log they check are Postgres behaviour.
"""

import asyncio
import json
import os
import uuid

import pytest
import pytest_asyncio

from dsp_decikit.services.live_sync import Rejected, check_message, stamp_ids

RULE_A = {"meta": {"name": "A"}, "rule": {"id": "a", "type": "leaf", "result_idx": -1}}
RULE_B = {"meta": {"name": "B"}, "rule": {"id": "b", "type": "leaf", "result_idx": -1}}


def flat(*rules):
    return {
        "type": "prioritized_flat_rule",
        "mode": "first_match",
        "parameters": {},
        "output": {"dtypes": [["decision", "String"]], "type_defs": {}, "data": [], "default": {"decision": "no"}},
        "rules": list(rules),
    }


# ---- message shape ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ops", [
    "not a list",
    [{"op": "move", "path": "/a"}],
    [{"op": "add", "path": "a", "value": 1}],
    [{"op": "add", "path": "/a"}],
    [{"op": "order", "path": "/rules", "value": "ab"}],
    [{"op": "remove", "path": 3}],
    [{"op": "replace", "path": "", "value": {}}],
])
def test_a_malformed_ops_message_is_rejected_whole(ops):
    with pytest.raises(Rejected):
        check_message({"type": "ops", "id": "m1", "base": 1, "ops": ops})


@pytest.mark.parametrize("message", [
    {"type": "ops", "id": "", "base": 1, "ops": []},
    {"type": "ops", "id": "m1", "base": -1, "ops": []},
    {"type": "ops", "id": "m1", "base": True, "ops": []},
    {"type": "ops", "id": "m1", "base": 1, "ops": [{"op": "remove", "path": "/a"}] * 5001},
])
def test_an_ops_message_without_an_id_with_a_bad_base_or_too_many_ops_is_rejected(message):
    with pytest.raises(Rejected):
        check_message(message)


def test_a_well_formed_ops_message_passes():
    check_message({"type": "ops", "id": "m1", "base": 3, "ops": [
        {"op": "replace", "path": "/rules/a/meta/name", "value": "x"},
        {"op": "remove", "path": "/rules/b"},
        {"op": "order", "path": "/rules", "value": ["b", "a"]},
    ]})


# ---- id stamping -----------------------------------------------------------------------------------------------

def test_stamping_gives_every_rule_node_an_id_and_keeps_existing_ones():
    doc = flat(
        {"meta": {"name": "A"}, "rule": {"type": "unary", "condition": {"type": "unary", "op": "<=", "feature": "x",
                                          "threshold": 1}, "then": {"type": "leaf", "result_idx": 0}}},
        RULE_B,
    )

    ops = stamp_ids(doc)

    from dsp_decikit.services.live_ops import apply
    stamped, dropped = apply(doc, ops)
    first = stamped["rules"][0]["rule"]
    assert dropped == [] and first["id"] and first["then"]["id"] and "id" not in first["condition"]
    assert stamped["rules"][1] == RULE_B


def test_stamping_a_document_whose_rules_all_have_ids_changes_nothing():
    assert stamp_ids(flat(RULE_A, RULE_B)) == []


# ---- the online rule, against Postgres -------------------------------------------------------------------------

db = pytest.mark.skipif(os.environ.get("LIVE_SYNC_DB") != "1", reason="needs the e2e stack's Postgres (LIVE_SYNC_DB=1)")


@pytest_asyncio.fixture
async def project():
    from sqlalchemy import select
    from dsp_decikit.db.session import db_session, engine
    from dsp_decikit.db.models import Project, ProjectType, User

    async with db_session() as session:
        user = User(email=f"{uuid.uuid4().hex[:10]}@example.com")
        session.add(user)
        await session.flush()
        row = Project(name="live", project_type=ProjectType.FLAT_RULE, project_data=flat(RULE_A, RULE_B),
                      created_by=user.id)
        session.add(row)
        await session.commit()
        project_id = row.id

    async def stored():
        async with db_session() as session:
            return (await session.execute(select(Project).where(Project.id == project_id))).scalar_one()

    yield project_id, stored
    await engine.dispose()


async def log(project_id):
    from sqlalchemy import select
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import ProjectOp

    async with db_session() as session:
        rows = await session.execute(select(ProjectOp).where(ProjectOp.project_id == project_id)
                                     .order_by(ProjectOp.version))
        return [(r.version, r.client_id, r.msg_id, r.ops) for r in rows.scalars()]


def name(rule, value):
    return {"op": "replace", "path": f"/rules/{rule}/meta/name", "value": value}


@db
@pytest.mark.asyncio
async def test_each_message_bumps_the_version_and_logs_the_ops_it_applied(project):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = project

    applied = await apply_message(project_id, "c1", "m1", 1, [name("a", "First")])

    assert (applied.version, applied.dropped) == (2, [])
    assert (await stored()).project_data["rules"][0]["meta"]["name"] == "First"
    assert await log(project_id) == [(2, "c1", "m1", [name("a", "First")])]


@db
@pytest.mark.asyncio
async def test_an_edit_inside_a_rule_another_client_replaced_since_my_base_is_dropped(project):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = project
    replaced = {"op": "replace", "path": "/rules/a/rule", "value": {"id": "a", "type": "leaf", "result_idx": 0}}
    await apply_message(project_id, "c1", "m1", 1, [replaced])

    inside = {"op": "replace", "path": "/rules/a/rule/result_idx", "value": -1}
    applied = await apply_message(project_id, "c2", "m2", 1, [inside, name("b", "Kept")])

    assert applied.dropped == [inside]
    data = (await stored()).project_data
    assert data["rules"][0]["rule"]["result_idx"] == 0 and data["rules"][1]["meta"]["name"] == "Kept"
    assert (await log(project_id))[-1][3] == [name("b", "Kept")]


@db
@pytest.mark.asyncio
async def test_my_own_earlier_messages_never_make_my_later_edits_stale(project):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = project
    await apply_message(project_id, "c1", "m1", 1, [{"op": "replace", "path": "/rules/a/rule",
                                                     "value": {"id": "a", "type": "leaf", "result_idx": 0}}])

    applied = await apply_message(project_id, "c1", "m2", 1, [{"op": "replace", "path": "/rules/a/rule/result_idx",
                                                               "value": -1}])

    assert applied.dropped == []
    assert (await stored()).project_data["rules"][0]["rule"]["result_idx"] == -1


@db
@pytest.mark.asyncio
async def test_the_same_field_edited_by_two_clients_keeps_the_last_writer(project):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = project

    await apply_message(project_id, "c1", "m1", 1, [name("a", "One")])
    applied = await apply_message(project_id, "c2", "m2", 1, [name("a", "Two")])

    assert applied.dropped == []
    assert (await stored()).project_data["rules"][0]["meta"]["name"] == "Two"


@db
@pytest.mark.asyncio
async def test_a_message_that_leaves_no_valid_document_is_rejected_and_nothing_is_stored(project):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = project

    with pytest.raises(Rejected):
        await apply_message(project_id, "c1", "m1", 1, [{"op": "replace", "path": "/type", "value": "ui-tree"}])

    row = await stored()
    assert (row.version, row.project_data["type"]) == (1, "prioritized_flat_rule")
    assert await log(project_id) == []


@db
@pytest.mark.asyncio
async def test_a_base_newer_than_the_project_is_rejected(project):
    from dsp_decikit.services.live_sync import apply_message
    project_id, _ = project

    with pytest.raises(Rejected):
        await apply_message(project_id, "c1", "m1", 5, [name("a", "x")])


@db
@pytest.mark.asyncio
async def test_a_rest_save_is_logged_as_one_root_replace(project):
    from dsp_decikit.services.live_sync import save_version
    from dsp_decikit.db.session import db_session
    project_id, stored = project
    document = flat(RULE_B)

    async with db_session() as session:
        version = await save_version(session, project_id, None, {"project_data": document}, "rest", "m1")
        await session.commit()

    assert version == 2 and (await stored()).project_data == document
    assert await log(project_id) == [(2, "rest", "m1", [{"op": "replace", "path": "", "value": document}])]


@db
@pytest.mark.asyncio
async def test_a_rest_save_that_only_renames_keeps_the_version_and_logs_nothing(project):
    from dsp_decikit.services.live_sync import save_version
    from dsp_decikit.db.session import db_session
    project_id, stored = project

    async with db_session() as session:
        assert await save_version(session, project_id, None, {"name": "Renamed"}, "rest", "m1") == 1
        await session.commit()

    assert ((await stored()).name, (await stored()).version) == ("Renamed", 1)
    assert await log(project_id) == []


@db
@pytest.mark.asyncio
async def test_two_writers_on_one_project_are_serialised_by_the_row_lock(project):
    from sqlalchemy import select
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import Project
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = project

    async with db_session() as holder:
        await holder.execute(select(Project).where(Project.id == project_id).with_for_update())
        first = asyncio.create_task(apply_message(project_id, "c1", "m1", 1, [name("a", "One")]))
        second = asyncio.create_task(apply_message(project_id, "c2", "m2", 1, [name("b", "Two")]))
        await asyncio.sleep(0.5)
        assert not first.done() and not second.done()  # both wait for the lock
        await holder.commit()
    done = await asyncio.gather(first, second)

    assert sorted(a.version for a in done) == [2, 3]
    data = (await stored()).project_data
    assert [r["meta"]["name"] for r in data["rules"]] == ["One", "Two"]
    assert [v for v, *_ in await log(project_id)] == [2, 3]


# ---- fan-out ---------------------------------------------------------------------------------------------------

class Socket:
    """A websocket that records what the hub sends; `on_send` runs before each send."""

    def __init__(self, on_send=None):
        self.sent, self.on_send = [], on_send

    async def send_text(self, text):
        if self.on_send:
            self.on_send()
        self.sent.append(json.loads(text))

    def versions(self):
        return [m["version"] for m in self.sent if m["type"] == "applied"]


@db
@pytest.mark.asyncio
async def test_a_client_joining_while_the_last_other_client_leaves_still_gets_new_versions(project):
    from dsp_decikit.services.live_hub import Client, Hub
    from dsp_decikit.services.live_sync import apply_message
    project_id, _ = project
    hub = Hub()
    first = Client(Socket(), "c1")
    await hub.join(project_id, first, 1)
    # The other client leaves while the joiner is being brought up to date (here: while it gets its snapshot).
    second = Client(Socket(on_send=lambda: hub.leave(project_id, first)), "c2")

    await hub.join(project_id, second, 0)
    await apply_message(project_id, "c3", "m1", 1, [name("a", "After the join")])
    hub.pump_soon(project_id)
    await asyncio.gather(*[r.pumping for r in hub.rooms.values() if r.pumping])

    assert second.ws.versions() == [2]


@db
@pytest.mark.asyncio
async def test_a_version_the_fan_out_failed_to_fetch_still_reaches_a_connected_client(project):
    from dsp_decikit.services.live_hub import Client, Hub
    from dsp_decikit.services.live_sync import apply_message
    project_id, _ = project
    hub = Hub()
    client = Client(Socket(), "c1")
    await hub.join(project_id, client, 1)
    fetch, failures = hub._send_rows, [ConnectionError("no connection from the pool")]

    async def flaky(*args):
        if failures:
            raise failures.pop()
        await fetch(*args)

    hub._send_rows = flaky
    await apply_message(project_id, "c2", "m1", 1, [name("a", "Fetched later")])
    hub.pump_soon(project_id)  # the only notification this version gets

    for _ in range(50):
        if client.ws.versions():
            break
        await asyncio.sleep(0.1)
    assert client.ws.versions() == [2]


@db
@pytest.mark.asyncio
async def test_after_the_validator_process_dies_the_next_version_is_validated_again(project):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from dsp_decikit.services.live_hub import Client, Hub
    from dsp_decikit.services.live_sync import apply_message
    project_id, _ = project
    hub = Hub()
    hub.validator = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    with pytest.raises(BrokenProcessPool):  # killed, as by the out-of-memory killer
        await asyncio.get_running_loop().run_in_executor(hub.validator, os._exit, 1)
    client = Client(Socket(), "c1")
    await hub.join(project_id, client, 1)
    await asyncio.gather(*[r.validating for r in hub.rooms.values() if r.validating], return_exceptions=True)

    await apply_message(project_id, "c2", "m1", 1, [name("a", "Validated")])
    hub.pump_soon(project_id)
    for _ in range(300):
        if any(m["type"] == "validation" and m["version"] == 2 for m in client.ws.sent):
            break
        await asyncio.sleep(0.1)
    hub.validator.shutdown()
    assert any(m["type"] == "validation" and m["version"] == 2 for m in client.ws.sent)
