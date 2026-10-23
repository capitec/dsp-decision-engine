"""Disk cost of the tree editor's op log for REST saves of the largest real flat-rule set (generated-521).

Creates the project, gives every rule node an id (as the editor does when it loads a document), then saves it SAVES
times over REST, renaming one rule between saves. Prints, per kind of log row, the stored size of `ops` and
`snapshot` (`pg_column_size`, after Postgres' TOAST compression) against their JSON text, and how much
`project_ops` grew (`pg_total_relation_size`, table + TOAST + indexes). COMPRESSION=lz4 sets both columns'
compression first (it applies to rows written after). Last, the size of the diff a save would log for a few
whole-list edits, against the whole document.

    BASE=http://localhost:9630 PG_PORT=9632 SAVES=200 uv run --no-project \
        --with-requirements ui/tree-editor/backend/requirements.txt --with-editable . --with ui/tree-editor/merge-core \
        --with polars python benchmarks/tree_editor_op_log_size.py
"""

import asyncio
import copy
import json
import os
import sys
import uuid
from pathlib import Path

import asyncpg
import httpx
from merge_core import apply, diff

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ui/tree-editor/e2e"), str(ROOT / "ui/tree-editor/backend")]
from real_trees import BY_ID  # noqa: E402
from dsp_decikit.services.live_sync import stamp_ids  # noqa: E402

BASE = os.environ.get("BASE", "http://localhost:9630")
PG = f"postgresql://postgres:postgres@localhost:{os.environ.get('PG_PORT', '9632')}/postgres"
SAVES = int(os.environ.get("SAVES", "200"))
COMPRESSION = os.environ.get("COMPRESSION")

ROWS = """
SELECT CASE WHEN version = 1 THEN 'created' WHEN snapshot IS NOT NULL THEN 'save + snapshot' ELSE 'save' END kind,
       count(*), avg(pg_column_size(ops))::int ops_stored, avg(octet_length(ops::text))::int ops_text,
       avg(pg_column_size(snapshot))::int snapshot_stored, avg(octet_length(snapshot::text))::int snapshot_text
FROM project_ops WHERE project_id = $1 GROUP BY 1 ORDER BY 1"""


def kb(n):
    return "-" if n is None else f"{n / 1024:.1f} KB"


def edited(doc, edit):
    doc = copy.deepcopy(doc)
    for i, rule in enumerate(doc["rules"]):
        edit(i, rule)
    return doc


def whole_list_edits(doc):
    """The diff a save logs, in JSON bytes, for one small edit and for edits that touch every rule."""
    unstamped = edited(doc, lambda i, r: r["rule"].pop("id"))
    cases = {
        "one rule renamed": (doc, edited(doc, lambda i, r: i == 7 and r["meta"].update(name="renamed"))),
        "rules reversed": (doc, {**doc, "rules": doc["rules"][::-1]}),
        "every rule renamed": (doc, edited(doc, lambda i, r: r["meta"].update(name=r["meta"]["name"] + " v2"))),
        "every rule given a new id": (doc, edited(doc, lambda i, r: r["rule"].update(id=str(uuid.uuid4())))),
        "ids stamped on a document without them": (unstamped, doc),
    }
    for label, (old, new) in cases.items():
        ops = diff(old, new)
        assert apply(old, ops)[0] == new
        print(f"  {label}: {len(ops)} ops, {kb(len(json.dumps(ops)))} (whole document {kb(len(json.dumps(new)))})")


async def main():
    db = await asyncpg.connect(PG)
    if COMPRESSION:
        await db.execute(f"ALTER TABLE project_ops ALTER COLUMN ops SET COMPRESSION {COMPRESSION}, "
                         f"ALTER COLUMN snapshot SET COMPRESSION {COMPRESSION}")
    doc = BY_ID["generated-521"].document
    doc = apply(doc, stamp_ids(doc))[0]
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as http:
        email = f"{uuid.uuid4().hex[:10]}@example.com"
        await http.post("/api/auth/basic/register", data={"email": email, "password": "Passw0rd!x", "name": "Bench"})
        if "session_id" not in http.cookies:
            await http.post("/api/auth/basic/login", data={"email": email, "password": "Passw0rd!x"})
        project = (await http.post("/api/v1/projects", json={"name": "op log size", "project_type": "flat_rule",
                                                              "project_data": doc})).json()
        before = await db.fetchval("SELECT pg_total_relation_size('project_ops')")
        version = project["version"]
        for i in range(SAVES):
            doc["rules"][i * 37 % len(doc["rules"])]["meta"]["name"] = f"save {i}"
            r = await http.put(f"/api/v1/projects/{project['id']}", json={"project_data": doc, "base_version": version})
            assert r.status_code == 200, r.text
            version = r.json()["version"]
    await db.execute("CHECKPOINT")
    grown = await db.fetchval("SELECT pg_total_relation_size('project_ops')") - before
    print(f"generated-521, {SAVES} REST saves, compression {COMPRESSION or 'default (pglz)'}; "
          f"document {kb(len(json.dumps(doc)))} as JSON")
    for row in await db.fetch(ROWS, uuid.UUID(project["id"])):
        print(f"  {row['kind']}: n={row['count']} ops {kb(row['ops_stored'])} stored / {kb(row['ops_text'])} text, "
              f"snapshot {kb(row['snapshot_stored'])} stored / {kb(row['snapshot_text'])} text")
    print(f"  project_ops grew {kb(grown)}: {kb(grown / SAVES)} per save")
    await db.close()
    print("diff a save would log:")
    whole_list_edits(doc)


asyncio.run(main())
