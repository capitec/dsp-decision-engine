"""Writing project versions: every document write locks the project row, bumps its version, logs the ops that made
the new version in `project_ops` and notifies the `project_ops` channel in the same transaction.

A live message (`apply_message`) follows the online rule: an op inside something another client changed since the
message's base is dropped, the rest apply, and the last writer wins per field. An offline message (edits made while
disconnected, sent as one batch) merges whole only when none of its paths overlaps what others changed since its base
and it brings no new validation error; otherwise it is rejected whole. Its validation runs without the row lock where
it can (see `_merge`). A REST save (`save_version`) is one root-replace op; only document changes make versions.
"""

import asyncio
import json
import logging
import time
import typing as t
import uuid
import weakref
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dsp_decikit.db.models.projects import Project, ProjectOp
from dsp_decikit.db.session import db_session
from dsp_decikit.services.documents import check_document, validation_errors
from dsp_decikit.services.live_ops import apply, conflicts, diff, rebase_orders, stale_ops

CHANNEL = "project_ops"
MAX_OPS = 5000
ATTEMPTS = 3
logger = logging.getLogger("uvicorn.error").getChild("live")
_merging: "weakref.WeakValueDictionary[uuid.UUID, asyncio.Lock]" = weakref.WeakValueDictionary()


class Rejected(Exception):
    """A message the server refuses whole; the reason (and an offline batch's conflicting paths) go to its author."""

    def __init__(self, reason: str, conflicts: t.Sequence[str] = ()):
        super().__init__(reason)
        self.conflicts = list(conflicts)


class StaleVersion(Exception):
    def __init__(self, current: int):
        self.current = current


@dataclass
class Applied:
    version: int
    ops: list
    dropped: list = field(default_factory=list)


def check_message(message: t.Any) -> None:
    """Raise Rejected unless `message` is a well-formed `ops` message (the op checks raise on malformed paths)."""
    if not isinstance(message, dict) or not isinstance(message.get("id"), str) or not 0 < len(message["id"]) <= 64:
        raise Rejected("a message needs an id")
    base = message.get("base")
    if not isinstance(base, int) or isinstance(base, bool) or base < 0:
        raise Rejected("base must be a version")
    ops = message.get("ops")
    if not isinstance(ops, list) or len(ops) > MAX_OPS:
        raise Rejected(f"ops must be a list of at most {MAX_OPS}")
    for op in ops:
        kind = op.get("op") if isinstance(op, dict) else None
        path = op.get("path") if kind else None
        if kind not in ("add", "replace", "remove", "order") or not isinstance(path, str) or not path.startswith("/"):
            raise Rejected(f"not an op: {op!r}")
        if kind != "remove" and "value" not in op:
            raise Rejected(f"{kind} needs a value: {path}")
        if kind == "order" and not (isinstance(op["value"], list) and all(isinstance(k, str) for k in op["value"])):
            raise Rejected(f"order needs a list of keys: {path}")
    parts = message.get("parts", [])
    if not isinstance(parts, list) or not all(
            isinstance(p, list) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], int)
            and not isinstance(p[1], bool) and p[1] >= 0 for p in parts) or sum(p[1] for p in parts) > len(ops):
        raise Rejected("parts must be [message id, op count] pairs covering the ops")
    if _has_nul(ops):
        # Postgres JSON cannot hold one: the write would fail and the client would send it again and again.
        raise Rejected("text cannot contain a NUL character")


def _has_nul(value: t.Any) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    return isinstance(value, list) and any(_has_nul(v) for v in value)


def _stamp(node: t.Any) -> None:
    # The flat-rule editor gives every rule node an id when it loads a document; the server does it once for all
    # clients, so two editors never stamp different ids on the same node.
    if not isinstance(node, dict):
        return
    if not node.get("id"):
        node["id"] = str(uuid.uuid4())
    kind = node.get("type")
    if kind in ("unary", "composite"):
        for child in (node.get("then"), node.get("otherwise")):
            if child:
                _stamp(child)
        if kind == "composite":
            for condition in node.get("conditions") or []:
                if isinstance(condition, dict) and condition.get("type") == "composite":
                    _stamp(condition)
    elif kind == "cases":
        for branch in node.get("branches") or []:
            _stamp(branch)


def stamp_ids(doc: t.Dict[str, t.Any]) -> list:
    """The ops that give every rule node of a flat-rule document an id (none when they all have one)."""
    rules = doc.get("rules")
    if not isinstance(rules, list):
        return []
    stamped = json.loads(json.dumps(rules))
    for rule in stamped:
        if isinstance(rule, dict):
            _stamp(rule.get("rule"))
    return diff({"rules": rules}, {"rules": stamped})


async def _locked(session: AsyncSession, project_id: uuid.UUID, lock: bool = True) -> Project:
    query = select(Project).where(Project.id == project_id)
    project = (await session.execute(query.with_for_update() if lock else query)).scalar_one_or_none()
    if project is None:
        raise Rejected("the project no longer exists")
    return project


async def _commit_version(session: AsyncSession, project: Project, ops: list, client_id: str, msg_id: str,
                          parts: t.Sequence[str] = (), **values) -> int:
    version = project.version + 1
    await session.execute(update(Project).where(Project.id == project.id)
                          .values(**values, version=version, updated_at=datetime.now(timezone.utc))
                          .execution_options(synchronize_session=False))
    await session.execute(insert(ProjectOp).values(project_id=project.id, version=version, client_id=client_id,
                                                   msg_id=msg_id, ops=ops, parts=list(parts)))
    # Delivered only when the transaction commits, so a listener never fetches a row it cannot see yet.
    await session.execute(select(func.pg_notify(CHANNEL, json.dumps({"p": str(project.id), "v": version}))))
    return version


async def save_version(session: AsyncSession, project_id: uuid.UUID, base_version: t.Optional[int],
                       changes: t.Dict[str, t.Any], client_id: str, msg_id: str) -> int:
    """Write `changes` (columns of the project); the caller commits. Only a document change makes a new version,
    logged as one root-replace op; a rename or move keeps the version and ignores `base_version`. Raises StaleVersion
    when a document save names a `base_version` the project has moved on from."""
    project = await _locked(session, project_id)
    if "project_data" in changes and base_version is not None and project.version != base_version:
        raise StaleVersion(project.version)
    if "project_data" not in changes:
        await session.execute(update(Project).where(Project.id == project_id)
                              .values(**changes, updated_at=datetime.now(timezone.utc))
                              .execution_options(synchronize_session=False))
        return project.version
    ops = [{"op": "replace", "path": "", "value": changes["project_data"]}]
    return await _commit_version(session, project, ops, client_id, msg_id, **changes)


def _unreachable(base: int) -> Rejected:
    return Rejected(f"the log no longer reaches back to version {base}; reload the project")


async def _since(session: AsyncSession, project: Project, base: int) -> list:
    if base > project.version:
        raise Rejected(f"base {base} is newer than the project (version {project.version})")
    rows = (await session.execute(select(ProjectOp.version, ProjectOp.client_id, ProjectOp.msg_id, ProjectOp.ops,
                                         ProjectOp.parts)
                                  .where(ProjectOp.project_id == project.id, ProjectOp.version > base)
                                  .order_by(ProjectOp.version))).all()
    if len(rows) != project.version - base:
        raise _unreachable(base)
    return rows


async def _document_at(session: AsyncSession, project_id: uuid.UUID, version: int) -> dict:
    # Replayed from the last whole-document write at or before `version` (a new project's first version is one).
    # ponytail: replays every version since that write; store a snapshot every N versions if replays get long.
    start = await session.scalar(select(func.max(ProjectOp.version)).where(
        ProjectOp.project_id == project_id, ProjectOp.version <= version, ProjectOp.ops[0]["path"].astext == ""))
    if start is None:
        raise _unreachable(version)
    rows = (await session.scalars(select(ProjectOp.ops).where(
        ProjectOp.project_id == project_id, ProjectOp.version.between(start, version))
        .order_by(ProjectOp.version))).all()
    return apply({}, [op for ops in rows for op in ops])[0]


def _error_key(error: dict) -> tuple:
    # Not `loc`: it holds rule positions, which a merge shifts. Not `input`: the value, which either side may edit.
    return error.get("type"), error.get("msg")


async def _new_errors(merged: dict, current: dict, own: t.Callable[[], dict], base: dict, validate) -> list:
    """The validation errors of `merged` beyond what one side had, or both sides added on their own, counted per
    (type, message); each document is validated only when still needed: validating a large one takes a few hundred
    ms."""
    async def counted(doc):
        return Counter(map(_error_key, await validate(doc)))

    errors = await validate(merged)
    excess = Counter(map(_error_key, errors))
    if excess:
        theirs = await counted(current)
        excess -= theirs
    if excess:
        mine = await counted(own())
        excess -= mine - theirs
    if excess:
        excess -= (theirs + mine - await counted(base)) - (theirs | mine)
    new = []
    for error in errors:
        if excess[_error_key(error)] > 0:
            excess[_error_key(error)] -= 1
            new.append(error)
    return new


async def _validate_here(doc: dict) -> list:
    return await asyncio.to_thread(validation_errors, doc)


async def apply_message(project_id: uuid.UUID, client_id: str, msg_id: str, base: int, ops: list,
                        offline: bool = False, validate: t.Optional[t.Callable[[dict], t.Awaitable[list]]] = None,
                        parts: t.Sequence = ()) -> Applied:
    """Apply one live message and commit it as the next version; a message already in the log is not applied again.

    Online, an op inside something another client changed since `base` is dropped. `offline` merges the batch whole
    or not at all (Rejected names the conflicting paths); `validate` (document to validation errors) checks the
    merge, in a thread when left out. `parts` (`[msg_id, op count]` in order) names earlier messages of this client
    the batch contains; the ops of one already logged are skipped. Raises Rejected (nothing stored) when the base is
    unknown or the result is not a document of the project's type.
    """
    if offline:
        return await _merge(project_id, client_id, msg_id, base, ops, parts, validate or _validate_here)
    started = time.perf_counter()
    async with db_session() as session:
        project = await _locked(session, project_id)
        locked = time.perf_counter()
        rows = await _since(session, project, base)
        logged = _logged(rows, client_id, msg_id)
        if logged:
            return logged
        ops = _unlogged(ops, parts, _logged_ids(rows, client_id))
        stale = stale_ops(ops, [op for row in rows if row.client_id != client_id for op in row.ops])
        fresh = [op for op in ops if not any(op is s for s in stale)]
        doc, failed = apply(project.project_data, fresh)
        applied = [op for op in fresh if not any(op is f for f in failed)]
        applying = time.perf_counter()
        _check(doc, project.project_type)
        checked = time.perf_counter()
        version = await _commit(session, project, doc, applied, client_id, msg_id, parts)
    done = time.perf_counter()
    logger.info("live apply project=%s version=%d offline=False ops=%d dropped=%d lock_ms=%.1f apply_ms=%.1f "
                "check_ms=%.1f persist_ms=%.1f total_ms=%.1f", project_id, version, len(ops), len(stale) + len(failed),
                (locked - started) * 1e3, (applying - locked) * 1e3, (checked - applying) * 1e3,
                (done - checked) * 1e3, (done - started) * 1e3)
    return Applied(version, applied, stale + failed)


async def _merge(project_id: uuid.UUID, client_id: str, msg_id: str, base: int, ops: list, parts: t.Sequence,
                 validate) -> Applied:
    # Validating a large document takes a few hundred ms, too long to hold the row lock: each attempt but the last
    # merges and validates from one snapshot without it, then commits only if no version came in meanwhile. The
    # last attempt holds the lock throughout, so a busy project still merges, and what is committed was validated.
    # Batches of one project wait for each other here, so they do not keep overtaking each other's attempts.
    started = time.perf_counter()
    async with _merging.setdefault(project_id, asyncio.Lock()):
        waited, base_doc = time.perf_counter(), None
        for attempt in range(1, ATTEMPTS + 1):
            last = attempt == ATTEMPTS
            async with db_session() as session:
                if not last:
                    await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
                project = await _locked(session, project_id, lock=last)
                read = locked = time.perf_counter()
                rows = await _since(session, project, base)
                logged = _logged(rows, client_id, msg_id)
                if logged:
                    return logged
                if base_doc is None:
                    base_doc = await _document_at(session, project_id, base)
                replayed = time.perf_counter()
                doc, applied, failed = await _merged(project, rows, base_doc, client_id, ops, parts, validate)
                validated = time.perf_counter()
                if last:
                    version = await _commit(session, project, doc, applied, client_id, msg_id, parts)
                    break
            async with db_session() as session:
                current = await _locked(session, project_id)
                locked = time.perf_counter()
                if current.version == project.version:
                    version = await _commit(session, current, doc, applied, client_id, msg_id, parts)
                    break
    done = time.perf_counter()
    logger.info("live apply project=%s version=%d offline=True ops=%d dropped=%d attempts=%d wait_ms=%.1f "
                "replay_ms=%.1f validate_ms=%.1f held_ms=%.1f total_ms=%.1f", project_id, version, len(applied),
                len(failed), attempt, (waited - started) * 1e3, (replayed - read) * 1e3, (validated - replayed) * 1e3,
                (done - locked) * 1e3, (done - started) * 1e3)
    return Applied(version, applied, failed)


async def _merged(project: Project, rows: list, base_doc: dict, client_id: str, ops: list, parts: t.Sequence,
                  validate) -> tuple:
    ops = _unlogged(ops, parts, _logged_ids(rows, client_id))
    found = conflicts(ops, [op for row in rows if row.client_id != client_id for op in row.ops], base_doc)
    if found:
        raise Rejected("someone else changed what your offline edits change", found)
    ops = rebase_orders(ops, base_doc, project.project_data)
    doc, failed = apply(project.project_data, ops)
    _check(doc, project.project_type)
    mine = [op for row in rows if row.client_id == client_id for op in row.ops] + ops
    try:
        new = await _new_errors(doc, project.project_data, lambda: apply(base_doc, mine)[0], base_doc, validate)
    except (asyncio.TimeoutError, BrokenProcessPool):
        raise Rejected("the server could not check your offline edits for errors in time, so they were not merged")
    if new:
        raise Rejected("merging your offline edits would add these errors: "
                       + "; ".join(sorted({e["msg"] for e in new})))
    return doc, [op for op in ops if not any(op is f for f in failed)], failed


def _logged(rows: list, client_id: str, msg_id: str) -> t.Optional[Applied]:
    # A message still in flight when its client reconnected can be logged on its own or inside a batch, and arrive
    # again either way: inside a later batch, or late on the socket that dropped.
    for row in rows:
        if row.client_id == client_id and (row.msg_id == msg_id or msg_id in row.parts):
            return Applied(row.version, row.ops)
    return None


def _logged_ids(rows: list, client_id: str) -> set:
    return {i for row in rows if row.client_id == client_id for i in [row.msg_id, *row.parts]}


def _check(doc: dict, project_type) -> None:
    try:
        check_document(doc, project_type)
    except ValueError as e:
        raise Rejected(str(e))


async def _commit(session: AsyncSession, project: Project, doc: dict, ops: list, client_id: str, msg_id: str,
                  parts: t.Sequence) -> int:
    version = await _commit_version(session, project, ops, client_id, msg_id, [p[0] for p in parts],
                                    project_data=doc)
    await session.commit()
    return version


def _unlogged(ops: list, parts: t.Sequence, logged: set) -> list:
    kept, at = [], 0
    for part_id, count in parts:
        if part_id not in logged:
            kept += ops[at:at + count]
        at += count
    return kept + ops[at:]


def first_version(project: Project) -> ProjectOp:
    """The log row of a new project's first version, so later versions can be rebuilt from the log."""
    return ProjectOp(project_id=project.id, version=1, client_id="server", msg_id=str(uuid.uuid4()),
                     ops=[{"op": "replace", "path": "", "value": project.project_data}])


async def stamp_project(project_id: uuid.UUID) -> None:
    """Give every rule node of the stored document an id, as one logged version, when some lack one."""
    async with db_session() as session:
        project = await _locked(session, project_id)
        ops = stamp_ids(project.project_data)
        if ops:
            doc, _ = apply(project.project_data, ops)
            await _commit_version(session, project, ops, "server", str(uuid.uuid4()), project_data=doc)
            await session.commit()
