"""Writing project versions: every document write locks the project row, bumps its version, logs the ops that made
the new version in `project_ops` and notifies the `project_ops` channel in the same transaction.

A live message (`apply_message`) follows the online rule: an op inside something another client changed since the
message's base is dropped, the rest apply, and the last writer wins per field. A REST save (`save_version`) is
one root-replace op; only document changes make versions.
"""

import json
import logging
import time
import typing as t
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dsp_decikit.db.models.projects import Project, ProjectOp
from dsp_decikit.db.session import db_session
from dsp_decikit.services.documents import check_document
from dsp_decikit.services.live_ops import apply, diff, stale_ops

CHANNEL = "project_ops"
MAX_OPS = 5000
logger = logging.getLogger("uvicorn.error").getChild("live")


class Rejected(Exception):
    """A message the server refuses whole; the reason goes back to its author."""


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


async def _locked(session: AsyncSession, project_id: uuid.UUID) -> Project:
    project = (await session.execute(select(Project).where(Project.id == project_id).with_for_update())).scalar_one_or_none()
    if project is None:
        raise Rejected("the project no longer exists")
    return project


async def _commit_version(session: AsyncSession, project: Project, ops: list, client_id: str, msg_id: str,
                          **values) -> int:
    version = project.version + 1
    await session.execute(update(Project).where(Project.id == project.id)
                          .values(**values, version=version, updated_at=datetime.now(timezone.utc))
                          .execution_options(synchronize_session=False))
    await session.execute(insert(ProjectOp).values(project_id=project.id, version=version, client_id=client_id,
                                                   msg_id=msg_id, ops=ops))
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


async def _concurrent(session: AsyncSession, project: Project, base: int, client_id: str) -> list:
    if base > project.version:
        raise Rejected(f"base {base} is newer than the project (version {project.version})")
    rows = (await session.execute(select(ProjectOp.client_id, ProjectOp.ops)
                                  .where(ProjectOp.project_id == project.id, ProjectOp.version > base)
                                  .order_by(ProjectOp.version))).all()
    if len(rows) != project.version - base:
        raise Rejected(f"the log no longer reaches back to version {base}; reload the project")
    return [op for row in rows if row.client_id != client_id for op in row.ops]


async def apply_message(project_id: uuid.UUID, client_id: str, msg_id: str, base: int, ops: list) -> Applied:
    """Apply one live message under the online rule and commit it as the next version.

    Raises Rejected (nothing stored) when the base is unknown or the result is not a document of the project's type.
    """
    started = time.perf_counter()
    async with db_session() as session:
        project = await _locked(session, project_id)
        locked = time.perf_counter()
        stale = stale_ops(ops, await _concurrent(session, project, base, client_id))
        fresh = [op for op in ops if not any(op is s for s in stale)]
        doc, failed = apply(project.project_data, fresh)
        applied = [op for op in fresh if not any(op is f for f in failed)]
        applying = time.perf_counter()
        try:
            check_document(doc, project.project_type)
        except ValueError as e:
            raise Rejected(str(e))
        checked = time.perf_counter()
        version = await _commit_version(session, project, applied, client_id, msg_id, project_data=doc)
        await session.commit()
    done = time.perf_counter()
    logger.info("live apply project=%s version=%d ops=%d dropped=%d lock_ms=%.1f apply_ms=%.1f check_ms=%.1f "
                "persist_ms=%.1f total_ms=%.1f", project_id, version, len(ops), len(stale) + len(failed),
                (locked - started) * 1e3, (applying - locked) * 1e3, (checked - applying) * 1e3,
                (done - checked) * 1e3, (done - started) * 1e3)
    return Applied(version, applied, stale + failed)


async def stamp_project(project_id: uuid.UUID) -> None:
    """Give every rule node of the stored document an id, as one logged version, when some lack one."""
    async with db_session() as session:
        project = await _locked(session, project_id)
        ops = stamp_ids(project.project_data)
        if ops:
            doc, _ = apply(project.project_data, ops)
            await _commit_version(session, project, ops, "server", str(uuid.uuid4()), project_data=doc)
            await session.commit()
