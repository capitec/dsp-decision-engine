# Live multi-session editing in the tree editor: how much does it cost?

Date: 2026-10-11 · Branch: `spike/live-collab` (not merged) · Scripts: `benchmarks/tree_editor_live_payload.py`,
`tree_editor_live_latency.py`, `tree_editor_live_server.py`, `tree_editor_live_offline.py`

## Purpose
Today a save PUTs the whole document and the last writer silently wins. Wanted: several sessions on one tree with
small continuous saves that every session sees, and offline edits that merge on reconnect when they do not conflict
(rejected and kept for the user otherwise). How hard is it, and would protobuf over websockets make it faster?

## Method
Three steps, each built test-first by a worker and reviewed by a second agent (each review found and fixed real
holes), merged into the spike branch separately so each can be taken alone (each builds on the previous one):
- **Step 0**: `projects.version` (document saves only); a save carries `base_version`; a stale save gets a 409 and
  the editor keeps the edits.
- **Step 1**: one websocket per project. The client diffs the document into path ops (lists of rules, nodes, edges
  and subtrees addressed by id, not index), pushes them 150 ms after the last edit, and applies everyone's ops in
  the server's version order. The server applies them under `SELECT ... FOR UPDATE`, logs them in `project_ops`
  and fans them out to the other backend nodes with Postgres `LISTEN/NOTIFY`. Online rule: the last writer wins per
  field, but an op inside something another session replaced or deleted is dropped and its author told. The
  flat-rule editor is wired; the node editor is not.
- **Step 2**: edits made while disconnected go up as one batch after the catch-up. The server merges it only if no
  path it touches overlaps one changed since the batch's base (parent, child or same path; a three-way rule for
  rule order), and the merged document has no validation error that neither side had. Otherwise the batch is
  refused and the client shows which rules conflicted and offers "Download my version".
- Measured: diff size per step, edit-to-other-browser latency (Playwright, N=50, one and two backend nodes),
  server time per message on the 521-rule sets, and wire size and codec time of 5,800 synthetic edits over the
  29 real trees as JSON, msgpack and protobuf.

## Result
**Size** (lines added/removed; product = shipped code without tests, migrations or generated types):

| step | product | tests | migrations | other | new infrastructure | new deps |
|---|---|---|---|---|---|---|
| 0 | +87/-24 (7 files) | +189/-30 | 1 column | 18 generated types | none | none |
| 1 | +1548/-114 (18 files; 543 of it the id-keyed diff/apply core, Python and TypeScript twins) | +1153 | `project_ops` | stack.sh 2 nodes, 2 benchmarks | websocket route, one LISTEN connection and one validator process per backend process | none |
| 2 | +381/-75 (11 files) | +610 | 2 (unique message index, `parts`) | 1 benchmark | none | none |

**Latency** (step 1, flat-rule editor; p50 / p95):

| | edit to shown in the other browser | of which the 150 ms debounce | socket send to shown |
|---|---|---|---|
| one node | 140 / 148 ms | 107 / 112 ms | 32 / 38 ms |
| two nodes | 140 / 156 ms | 107 / 115 ms | 33 / 41 ms |

- Server, 2-rule document: 15 ms per message, mostly the commit. 521-rule sets (126 to 194 KB): 55 to 60 ms
  round trip (lock+load 10, apply 9 to 10, JSONB rewrite and commit 21, auth 10).
- `validation_errors()` takes 190 to 230 ms on 521 rules. In a thread it starved the event loop through the GIL
  (round trip 250 ms), so it runs in its own process, for the latest version only, off the write path.
- An offline batch of 50 edits on 521 rules merges in 370 to 460 ms, almost all of it validation (216 to 323 ms)
  and log replay (70 ms), while holding the project's row lock. With 10 such batches at once, an online editor's
  round trip went from 63 ms to 1.5 s (p50).

**Protobuf**: no. An edit message is 130 to 250 bytes in every format, and encoding plus decoding costs 3 to 14
microseconds, well under 1% of a 1 ms local round trip. After permessage-deflate every format is 53 to 55 bytes.
Protobuf with `google.protobuf.Value` is worse on snapshots (23% bigger, 6 to 10 times slower to encode and decode)
and turned 276 of 11,629 messages' integers into floats. The cost a user sees is the 150 ms debounce, then render
and commit. The one large message is the rule-order op, which lists every rule id: 21 KB for a move on 521 rules.
A `move {key, after}` op would fix it; no format does.

**Multiple nodes**: the cache is per process (`InMemoryProvider`), so with two nodes and no sticky routing a login
made on node A gets a 401 on node B. `persisted-cache` fixes logins, but no node's memory layer is ever invalidated:
a session logged out on A stays valid on B, and B can expire a session that A renewed. Permissions are not cached.
OIDC logins that start on one node and return to the other fail.

**Risks found and fixed by the reviews**: a client joining a room as it emptied got nothing, ever; a failed fan-out
fetch lost a version; one dead validator stopped validation for good; a NUL character jammed a client in a resend
loop; a message in flight across a reconnect, or folded into an offline batch, applied twice (now deduplicated by
message id); "same value is no conflict" accepted a lost edit; a second disconnect gave offline edits too new a base.

**Risks still open** (after `collab/ship`; the rest of the spike's list is fixed, see below):
- `result_idx` points at output rows by index, so a merge can keep two edits that are each valid but point at the
  wrong row together.
- Dangling parameter references are not reported by validation at all.

Fixed on `collab/ship`: a message that fails every time (over 4 MiB, a server error) is handed back after three
tries through the offline banner (a dropped connection or a database outage is never counted); messages Postgres or
the core would refuse (NUL, lone surrogates, NaN, nesting past 64) are rejected up front; socket edits renew the
session by the HTTP rule and a sign-out, expiry or lost access closes the socket saying why (the editor stops and
keeps the edits until the user signs in again); one slow socket is dropped after 5 s instead of holding the fan-out;
a hung validator times out after 30 s and is replaced, and offline merges validate without the row lock
(`tree-editor-offline-merge-lock.md`); validation errors are counted, not a set; the node editor edits live.

Fixed since: both editors share one live hook, so the node editor also stops on a sign-out, expiry or lost access,
resumes after the session dialog and hands back edits the server fails on three times; the e2e smoke set and
`test_sessions.py` pass with the flag on (two nodes) as well as off.

Closed since, on `collab/ship`: the op log snapshots the document every 100 versions and is trimmed by count and age
(a rebuild replays at most 100 versions), and a migration logs the current version of projects created before step
2, so their offline batches merge. On the 521-rule set (123 KB) rebuilding a version takes 19 ms from its snapshot
and 29 ms 99 versions after it (loading the snapshot dominates; a version replays in about 0.1 ms); a write that
stores a snapshot and trims takes 53 to 55 ms against 39 ms (p50) for one that does not; the trim of a 6,200-version
log to 5,001 rows takes 18 ms. The trim looks for its whole row only below the count or age bound: on a log of
5,100 REST saves of 521 rules (1.2 GB, every row a whole document) filtering every row for being whole took 4.3 s
under the row lock, the bounded lookup 17 ms.

## Recommendation

| step | verdict | why |
|---|---|---|
| 0 version + 409 | **merge** | about 90 product lines, no infrastructure; ends silent overwrites from two tabs. Decide whether API PUTs that change `project_data` must send `base_version` (today they still overwrite). |
| 1 live sync | **merge later** | It works (convergent, 140 ms to the other browser, two nodes via LISTEN/NOTIFY), but it is about 1,550 product lines and new moving parts. Before it ships: wire the node editor (about 150 to 200 lines), fix the shared session store, cap repeated failures of one message, renew the session over the socket, and add log retention. Keep it behind `DSP_LIVE_COLLAB` until then. |
| 2 offline merge | **do not merge (yet)** | Step 1 alone already resends edits after a short drop under the online rule. Step 2 adds about 380 product lines, holds the row lock for up to half a second on big trees, needs log replay or snapshots, and its validation check is a heuristic. Revisit if users lose real work to long disconnects. |
