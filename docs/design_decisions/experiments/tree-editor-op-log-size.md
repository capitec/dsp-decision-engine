# Tree editor: what the op log costs on disk

Date: 2026-10-23 · Branch: `live/diff-saves` · Script: `benchmarks/tree_editor_op_log_size.py`

## Purpose
A REST save was logged as one root-replace op holding the whole document. known-limitations estimated 1.2 GB for a
heavily saved project (5000 saves of about 240 KB, uncompressed). What does Postgres really store per save, and what
does logging a save as `merge_core.diff(stored, saved)` change? Does `lz4` column compression help on top?

## Method
- One-node stack (`stack.sh`, Postgres 17.11 in podman; the compose file also runs `postgres:17`).
- generated-521 with every rule node given an id, as the editor does on load (177.6 KB as JSON). Created over REST,
  then saved 200 times over REST, one rule renamed between saves.
- Per log row: `pg_column_size(ops)`, `pg_column_size(snapshot)` (as stored, after TOAST compression) against
  `octet_length(...::text)`; `pg_total_relation_size('project_ops')` (table, TOAST and indexes) before and after
  the run, after a `CHECKPOINT`. Run with each column's compression set to `pglz` (the default) and to `lz4`.
- The diff a save would log for edits that touch every rule, computed in the script (JSON bytes).

## Result
Per save, 200 saves:

| | `ops` stored / text | snapshot (every 100th) stored / text | `project_ops` growth per save |
|---|---|---|---|
| before (root replace), pglz | 67.8 KB / 177.2 KB | 68.0 KB / 177.4 KB | 71.5 to 72.0 KB |
| before, lz4 | 73.6 KB / 177.2 KB | 73.9 KB / 177.4 KB | 78.3 KB |
| after (diff), pglz | 115 B / 104 B | 68.1 KB / 177.4 KB | 1.1 KB |
| after, lz4 | 115 B / 104 B | 73.8 KB / 177.4 KB | 1.2 KB |

- Before: TOAST compressed a whole document 2.6 times, so 5000 saves cost about 350 MB per project, not 1.2 GB.
- After: a one-rule save is a 115-byte row, too small to be compressed; the cost per save is the row, its index
  entries and a hundredth of a snapshot. 5000 saves cost about 5.5 MB. The first version stays whole (67 KB).
- `lz4` stores the same JSON 8% bigger than `pglz` (it trades size for speed); the diff rows are not compressed at
  all. Not added.
- Diffing the 521-rule document takes 17.7 ms (p50, n=30), under the row lock of the save.

The diff a save logs for edits that touch every rule (whole document 177.6 KB):

| edit | ops | JSON |
|---|---|---|
| one rule renamed | 1 | 0.1 KB |
| rules reversed | 1 (an order op of every id) | 20.4 KB |
| every rule renamed | 521 | 53.5 KB |
| ids stamped on a document without them | 1 (the rule list replaced) | 177.5 KB |
| every rule given a new id | 1042 (each removed, each added) | 254.8 KB |

Only a save that re-keys every rule logs more than the whole document (1.4 times); the editors keep rule ids and
stamp only missing ones.
