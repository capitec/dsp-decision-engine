"""Writing project versions: every document write locks the project row, bumps its version, logs the ops that made
the new version in `project_ops` and notifies the `project_ops` channel in the same transaction.

A live message (`apply_message`) follows the online rule: an op inside something another client changed since the
message's base is dropped, the rest apply, and the last writer wins per field. An offline message (edits made while
disconnected, sent as one batch) merges whole only when none of its paths overlaps what others changed since its base
and it brings no new validation error; otherwise it is rejected whole. Its validation runs without the row lock where
it can (see `_merge`). A REST save (`save_version`) is logged as the diff from the stored document, so it conflicts
only with edits that overlap what it changed; only document changes make versions.

Every `SNAPSHOT_EVERY` versions a log row also stores the whole document, and the log is trimmed to the versions
`settings.live_log_keep_versions` and `live_log_keep_days` keep, from a row holding a whole document on, so any
kept version is rebuilt by replaying at most `SNAPSHOT_EVERY` versions.
"""

import asyncio
import json
import logging
import math
import time
import typing as t
import uuid
import weakref
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dsp_decikit.db.models.projects import Project, ProjectOp
from dsp_decikit.db.session import db_session
from dsp_decikit.services.documents import check_document, validation_errors
from dsp_decikit.services.live_ops import apply, conflicts, diff, rebase_orders, stale_ops, touched
from dsp_decikit.settings import settings

CHANNEL = "project_ops"
MAX_OPS = 5000
# A rebuild replays at most this many versions; each snapshot is a whole document (120 to 190 KB on 521 rules).
SNAPSHOT_EVERY = 100
# The client ids a REST save and the server log under; a live client may not take them.
REST = "rest"
RESERVED_CLIENT_IDS = {REST, "server"}
# The merge core parses at most 128 levels; real documents nest about a dozen.
MAX_DEPTH = 64
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
    reason = _unstorable([message["id"], parts], 0) or next(
        filter(None, (_unstorable(op, op["path"].count("/")) for op in ops)), None)
    if reason:
        # Postgres or the merge core would refuse it on every try, and the client would send it again and again.
        raise Rejected(reason)


def _unstorable(value: t.Any, depth: int) -> t.Optional[str]:
    if depth > MAX_DEPTH:
        return f"edits cannot nest deeper than {MAX_DEPTH} levels"
    if isinstance(value, str):
        if "\x00" in value:
            return "text cannot contain a NUL character"
        try:
            value.encode()
        except UnicodeEncodeError:
            return "text cannot contain a lone surrogate (broken Unicode)"
    elif isinstance(value, float) and not math.isfinite(value):
        return "numbers must be finite"
    elif isinstance(value, (dict, list)):
        for item in [*value, *value.values()] if isinstance(value, dict) else value:
            reason = _unstorable(item, depth + 1)
            if reason:
                return reason
    return None


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
    snapshot = values["project_data"] if version % SNAPSHOT_EVERY == 0 else None
    await session.execute(update(Project).where(Project.id == project.id)
                          .values(**values, version=version, updated_at=datetime.now(timezone.utc))
                          .execution_options(synchronize_session=False))
    await session.execute(insert(ProjectOp).values(project_id=project.id, version=version, client_id=client_id,
                                                   msg_id=msg_id, ops=ops, parts=list(parts), snapshot=snapshot))
    if snapshot is not None:
        await _trim(session, project.id, version)
    # Delivered only when the transaction commits, so a listener never fetches a row it cannot see yet.
    await session.execute(select(func.pg_notify(CHANNEL, json.dumps({"p": str(project.id), "v": version}))))
    return version


async def save_version(session: AsyncSession, project_id: uuid.UUID, base_version: t.Optional[int],
                       changes: t.Dict[str, t.Any], client_id: str, msg_id: str) -> int:
    """Write `changes` (columns of the project); the caller commits. Only a document change makes a new version,
    logged as the diff from the stored document (no ops when it is the same); a rename or move keeps the version and
    ignores `base_version`. Raises StaleVersion when a document save names a `base_version` the project has moved on
    from."""
    project = await _locked(session, project_id)
    if "project_data" in changes and base_version is not None and project.version != base_version:
        raise StaleVersion(project.version)
    if "project_data" not in changes:
        await session.execute(update(Project).where(Project.id == project_id)
                              .values(**changes, updated_at=datetime.now(timezone.utc))
                              .execution_options(synchronize_session=False))
        return project.version
    try:
        ops = diff(project.project_data, changes["project_data"])
    except ValueError:  # nested deeper than the core parses: still saved, logged whole as before
        ops = [{"op": "replace", "path": "", "value": changes["project_data"]}]
    return await _commit_version(session, project, ops, client_id, msg_id, **changes)


def _whole(project_id: uuid.UUID):
    # The rows a document can be rebuilt from without the rows before them: snapshots, a new project's first version
    # and rows logged before REST saves were diffs.
    return (ProjectOp.project_id == project_id) & (ProjectOp.snapshot.is_not(None)
                                                   | (ProjectOp.ops[0]["path"].astext == ""))


async def _trim(session: AsyncSession, project_id: uuid.UUID, version: int) -> None:
    # Under the project's row lock, so no write, catch-up check or offline replay of this project runs meanwhile on
    # the server path; a reader outside the lock (a joining client) sees a gap and gets a snapshot.
    # Bounded by version first, so only the rows from the bound back are read for being whole (that reads their ops).
    aged = select(func.max(ProjectOp.version)).where(
        ProjectOp.project_id == project_id,
        ProjectOp.created_at < func.now() - timedelta(days=settings.live_log_keep_days)).scalar_subquery()
    oldest = select(func.max(ProjectOp.version)).where(
        _whole(project_id), ProjectOp.version <= func.greatest(version - settings.live_log_keep_versions, aged)
    ).scalar_subquery()
    await session.execute(delete(ProjectOp).where(ProjectOp.project_id == project_id, ProjectOp.version < oldest))


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
    # Replayed from the last whole document at or before `version` (a new project's first version is one).
    start = (await session.execute(select(ProjectOp.version, ProjectOp.snapshot).where(
        _whole(project_id), ProjectOp.version <= version).order_by(ProjectOp.version.desc()).limit(1))).first()
    if start is None:
        raise _unreachable(version)
    first = start.version + (start.snapshot is not None)
    rows = (await session.scalars(select(ProjectOp.ops).where(
        ProjectOp.project_id == project_id, ProjectOp.version.between(first, version))
        .order_by(ProjectOp.version))).all()
    if len(rows) != version - first + 1:
        raise _unreachable(version)  # a write that was not logged
    return apply(start.snapshot or {}, [op for ops in rows for op in ops])[0]


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

    Online, an op inside something another client changed since `base` is dropped, and so is one on, above or
    inside anything a REST save changed since then. `offline` merges the batch whole
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
        # A REST save's caller was told it saved, so an edit made before it never overwrites what it changed.
        saved = set(touched([op for row in rows if row.client_id == REST for op in row.ops]))
        above = {p for path in saved for p in _prefixes(path)}
        stale += [op for op in ops if not any(op is s for s in stale)
                  and ((path := touched([op])[0]) in above or any(p in saved for p in _prefixes(path)))]
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


def _prefixes(path: str) -> list:
    # "" and every ancestor of a JSON pointer, then the pointer itself; split by segment, as the core compares paths.
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


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
