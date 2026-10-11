"""The offline rule through the server path, against a real Postgres (TREE_EDITOR_DB=1, as `test_live_sync.py`):
a batch of edits made while disconnected merges whole or is rejected whole, and a message is never applied twice.
"""

import asyncio

import pytest
import pytest_asyncio

from dsp_decikit.services.live_sync import Rejected, check_message
from tests.test_live_sync import RULE_A, RULE_B, db, flat, log, name, project  # noqa: F401


# ---- the offline rule, against Postgres ---------------------------------------------------------------------------

RULE_C = {"meta": {"name": "C"}, "rule": {"id": "c", "type": "leaf", "result_idx": -1}}


@pytest_asyncio.fixture
async def offline(project):
    """A project of three rules and a parameter `limit`, logged from version 1 like one the API creates."""
    from sqlalchemy import update
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import Project, ProjectOp
    project_id, stored = project
    doc = {**flat(RULE_A, RULE_B, RULE_C), "parameters": {"limit": {"type": "Float", "default_value": 5}}}
    async with db_session() as session:
        await session.execute(update(Project).where(Project.id == project_id).values(project_data=doc))
        session.add(ProjectOp(project_id=project_id, version=1, client_id="server", msg_id="created",
                              ops=[{"op": "replace", "path": "", "value": doc}]))
        await session.commit()
    yield project_id, stored


async def merge(project_id, ops, base=1, client="offline", msg="batch"):
    from dsp_decikit.services.live_sync import apply_message
    return await apply_message(project_id, client, msg, base, ops, offline=True)


async def refused(project_id, ops, base=1) -> Rejected:
    with pytest.raises(Rejected) as e:
        await merge(project_id, ops, base)
    return e.value


def order(data):
    return [r["rule"]["id"] for r in data["rules"]]


@db
@pytest.mark.asyncio
async def test_offline_edits_to_other_rules_merge(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [name("a", "Online")])

    applied = await merge(project_id, [name("b", "Offline"), name("c", "Offline too")])

    assert (applied.version, applied.dropped) == (3, [])
    assert names(await stored()) == ["Online", "Offline", "Offline too"]


@db
@pytest.mark.asyncio
async def test_an_offline_edit_to_a_field_someone_changed_rejects_the_whole_batch_naming_the_path(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [name("a", "Online")])

    error = await refused(project_id, [name("a", "Offline"), name("b", "Offline")])

    assert error.conflicts == ["/rules/a/meta/name"]
    row = await stored()
    assert (row.version, names(row)) == (2, ["Online", "B", "C"])


@db
@pytest.mark.asyncio
@pytest.mark.parametrize("theirs, mine", [
    ([{"op": "remove", "path": "/rules/a"}], [name("a", "Offline")]),
    ([name("a", "Online")], [{"op": "remove", "path": "/rules/a"}]),
], ids=["they deleted the rule I edited", "I deleted the rule they edited"])
async def test_a_deleted_rule_against_an_edit_inside_it_rejects_either_way(offline, theirs, mine):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, theirs)

    error = await refused(project_id, mine)

    assert error.conflicts == [mine[0]["path"]]
    assert (await stored()).version == 2


def added(key):
    return {"op": "add", "path": f"/rules/{key}", "value": {"meta": {"name": key.upper()},
                                                            "rule": {"id": key, "type": "leaf", "result_idx": -1}}}


@db
@pytest.mark.asyncio
async def test_rules_added_on_both_sides_are_all_kept(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [added("d")])

    await merge(project_id, [added("e"), {"op": "order", "path": "/rules", "value": ["a", "e", "b", "c"]}])

    assert order((await stored()).project_data) == ["a", "e", "b", "c", "d"]


@db
@pytest.mark.asyncio
async def test_reordering_differently_on_both_sides_rejects(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [{"op": "order", "path": "/rules", "value": ["c", "b", "a"]}])

    error = await refused(project_id, [{"op": "order", "path": "/rules", "value": ["b", "a", "c"]}])

    assert error.conflicts == ["/rules/#order"]
    assert order((await stored()).project_data) == ["c", "b", "a"]


@db
@pytest.mark.asyncio
async def test_an_offline_insert_keeps_the_order_someone_else_gave_the_rules(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [{"op": "order", "path": "/rules", "value": ["c", "b", "a"]}])

    await merge(project_id, [added("d"), {"op": "order", "path": "/rules", "value": ["a", "d", "b", "c"]}])

    assert order((await stored()).project_data) == ["c", "b", "a", "d"]


USES_LIMIT = {"op": "replace", "path": "/rules/b/rule", "value": {
    "id": "b", "type": "unary", "then": {"id": "bt", "type": "leaf", "result_idx": -1},
    "condition": {"op": "<=", "feature": "amount", "threshold": {"param": "limit"}}}}
LIMIT_TEXT = {"op": "replace", "path": "/parameters/limit/default_value", "value": "abc"}


@db
@pytest.mark.asyncio
async def test_a_merge_that_makes_a_validation_error_neither_side_had_is_rejected(offline):
    # Their default is fine while no rule reads the parameter; my rule reads it as a number.
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [LIMIT_TEXT])

    error = await refused(project_id, [USES_LIMIT])

    assert error.conflicts == [] and "limit" in str(error)
    assert (await stored()).version == 2


@db
@pytest.mark.asyncio
async def test_a_validation_error_my_own_document_already_had_does_not_block_the_merge(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "online", "m1", 1, [name("a", "Online")])

    await merge(project_id, [LIMIT_TEXT, USES_LIMIT])

    assert (await stored()).version == 3


def between(key, **bounds):
    # Either bound alone is valid; a `between` with neither is an error whose message names no rule.
    return {"op": "replace", "path": f"/rules/{key}/rule", "value": {
        "id": key, "type": "unary", "then": {"id": f"{key}t", "type": "leaf", "result_idx": -1},
        "condition": {"op": "between", "feature": "amount", **bounds}}}


def unbound(key, bound):
    return {"op": "remove", "path": f"/rules/{key}/rule/condition/{bound}"}


@db
@pytest.mark.asyncio
async def test_a_merge_that_repeats_an_error_the_document_already_has_is_rejected(offline):
    # Rule c already has the error; their edit and mine each leave rule a valid, together they leave it without bounds.
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "setup", "m1", 1, [between("a", min=1, max=10), between("c")])
    await apply_message(project_id, "online", "m2", 2, [unbound("a", "min")])

    error = await refused(project_id, [unbound("a", "max")], base=2)

    assert error.conflicts == [] and "min or max" in str(error)
    assert (await stored()).version == 3


@db
@pytest.mark.asyncio
async def test_the_same_error_brought_by_each_side_on_its_own_does_not_block_the_merge(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "setup", "m1", 1, [between("a", min=1), between("b", min=1)])
    await apply_message(project_id, "online", "m2", 2, [unbound("a", "min")])

    await merge(project_id, [unbound("b", "min")], base=2)

    assert (await stored()).version == 4


@db
@pytest.mark.asyncio
async def test_an_offline_base_the_log_no_longer_reaches_is_rejected(project):
    # `project` has no logged first version, as when the log was trimmed or the project predates it.
    project_id, stored = project

    error = await refused(project_id, [name("a", "Offline")])

    assert "log" in str(error) and (await stored()).version == 1


@db
@pytest.mark.asyncio
async def test_a_message_applied_before_the_connection_dropped_and_sent_again_is_applied_once(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    first = await apply_message(project_id, "c1", "m1", 1, [added("d")])

    again = await apply_message(project_id, "c1", "m1", 1, [added("d")], offline=True)

    assert again.version == first.version == 2
    assert (await stored()).version == 2 and order((await stored()).project_data) == ["a", "b", "c", "d"]
    assert [v for v, *_ in await log(project_id)] == [1, 2]


def names(row):
    return [r["meta"]["name"] for r in row.project_data["rules"]]


# ---- a message resent while its first send was still in flight -------------------------------------------------

@db
@pytest.mark.asyncio
async def test_a_message_sent_twice_while_the_first_waits_for_the_lock_is_applied_once(offline):
    from sqlalchemy import select
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import Project
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline

    async with db_session() as holder:
        await holder.execute(select(Project).where(Project.id == project_id).with_for_update())
        sends = [asyncio.create_task(apply_message(project_id, "c1", "m1", 1, [added("d")])) for _ in range(2)]
        await asyncio.sleep(0.3)
        await holder.commit()
    done = await asyncio.gather(*sends)

    assert [a.version for a in done] == [2, 2]
    assert [v for v, *_ in await log(project_id)] == [1, 2]


@db
@pytest.mark.asyncio
async def test_a_resent_message_does_not_overwrite_an_edit_made_after_it_was_applied(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "c1", "m1", 1, [name("a", "Mine")])
    await apply_message(project_id, "c2", "m2", 2, [name("a", "Theirs")])

    await apply_message(project_id, "c1", "m1", 1, [name("a", "Mine")])

    assert names(await stored())[0] == "Theirs"


@db
@pytest.mark.asyncio
async def test_an_offline_batch_skips_the_messages_it_contains_that_were_applied_meanwhile(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "c1", "m1", 1, [name("a", "Mine")])
    await apply_message(project_id, "c2", "m2", 2, [name("a", "Theirs")])

    await apply_message(project_id, "c1", "batch", 1, [name("a", "Mine"), name("b", "Offline")], offline=True,
                        parts=[["m1", 1], ["m3", 1]])

    assert names(await stored())[:2] == ["Theirs", "Offline"]


@db
@pytest.mark.asyncio
async def test_a_message_that_arrives_after_the_offline_batch_containing_it_is_not_applied_again(offline):
    # Sent on the old socket before the drop and delivered late, after the batch that resent it was merged.
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    await apply_message(project_id, "c1", "batch", 1, [name("a", "Mine"), name("b", "Offline")], offline=True,
                        parts=[["m1", 1], ["m2", 1]])
    await apply_message(project_id, "c2", "m3", 2, [name("a", "Theirs")])

    late = await apply_message(project_id, "c1", "m1", 1, [name("a", "Mine")])

    assert late.version == 2 and names(await stored())[0] == "Theirs"
    assert [v for v, *_ in await log(project_id)] == [1, 2, 3]


@db
@pytest.mark.asyncio
async def test_a_batch_folded_into_a_later_batch_is_applied_once_whichever_arrives_first(offline):
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    first = [name("a", "Mine")]
    await apply_message(project_id, "c1", "batch1", 1, first, offline=True, parts=[["m1", 1]])
    await apply_message(project_id, "c2", "m3", 2, [name("a", "Theirs")])

    # The second batch resends the first one's message (its ack was lost) plus a newer edit.
    await apply_message(project_id, "c1", "batch2", 1, first + [name("b", "Later")], offline=True,
                        parts=[["m1", 1], ["m2", 1]])

    assert names(await stored())[:2] == ["Theirs", "Later"]


@db
@pytest.mark.asyncio
async def test_the_log_refuses_a_second_row_for_the_same_message(offline):
    from sqlalchemy.exc import IntegrityError
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import ProjectOp
    project_id, _ = offline

    with pytest.raises(IntegrityError):
        async with db_session() as session:
            session.add(ProjectOp(project_id=project_id, version=2, client_id="server", msg_id="created", ops=[]))
            await session.commit()


@pytest.mark.parametrize("parts", ["x", [["m1"]], [["m1", -1]], [[1, 1]], [["m1", 3]]])
def test_offline_parts_that_do_not_describe_the_ops_are_rejected(parts):
    with pytest.raises(Rejected):
        check_message({"type": "ops", "id": "b", "base": 1, "offline": True, "parts": parts,
                       "ops": [name("a", "x"), name("b", "y")]})


# ---- the row lock while a batch is validated ---------------------------------------------------------------------

async def row_locked(project_id) -> bool:
    from sqlalchemy import select
    from dsp_decikit.db.session import db_session
    from dsp_decikit.db.models import Project
    async with db_session() as session:
        free = await session.scalar(select(Project.id).where(Project.id == project_id).with_for_update(skip_locked=True))
    return free is None


@db
@pytest.mark.asyncio
async def test_an_edit_saved_while_an_offline_batch_is_validated_is_validated_with_it(offline):
    from dsp_decikit.services.documents import validation_errors
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    saved = []

    async def validate(doc):
        if not saved:
            assert not await row_locked(project_id)
            saved.append(await apply_message(project_id, "online", "m1", 1, [LIMIT_TEXT]))
        return validation_errors(doc)

    with pytest.raises(Rejected) as error:
        await apply_message(project_id, "offline", "batch", 1, [USES_LIMIT], offline=True, validate=validate)

    assert "limit" in str(error.value) and saved[0].version == 2
    assert (await stored()).version == 2


@db
@pytest.mark.asyncio
async def test_an_offline_batch_overtaken_by_an_edit_while_it_was_validated_merges_on_top_of_it(offline):
    from dsp_decikit.services.documents import validation_errors
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    validated = []

    async def validate(doc):
        if not validated:  # it would wait for ever if the merge held the row lock
            await asyncio.wait_for(apply_message(project_id, "online", "m1", 1, [name("a", "Online")]), 5)
        validated.append(doc)
        return validation_errors(doc)

    applied = await apply_message(project_id, "offline", "batch", 1, [name("b", "Offline")], offline=True,
                                  validate=validate)

    row = await stored()
    assert applied.version == row.version == 3 and names(row) == ["Online", "Offline", "C"]
    assert row.project_data == validated[-1]


@db
@pytest.mark.asyncio
async def test_an_offline_batch_overtaken_on_every_attempt_merges_under_the_row_lock(offline):
    from dsp_decikit.services.documents import validation_errors
    from dsp_decikit.services.live_sync import apply_message
    project_id, stored = offline
    validated = []

    async def validate(doc):
        validated.append(doc)
        if not await row_locked(project_id):
            await apply_message(project_id, "online", f"m{len(validated)}", 1, [name("a", f"Online {len(validated)}")])
        return validation_errors(doc)

    applied = await apply_message(project_id, "offline", "batch", 1, [name("b", "Offline")], offline=True,
                                  validate=validate)

    row = await stored()
    assert applied.version == row.version and row.project_data == validated[-1]
    assert names(row)[1] == "Offline"

