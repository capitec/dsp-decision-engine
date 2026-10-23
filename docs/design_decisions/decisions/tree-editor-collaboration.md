# Tree editor collaboration

Read this before changing how the tree editor saves, syncs or merges a project document, its websocket, the op log or
`merge-core/`: it records why each is shaped as it is. Measurements:
[live collab](../experiments/tree-editor-live-collab.md), [merge core](../experiments/tree-editor-merge-core.md).

## A document save names the version it was made on

- A save sends `base_version`; a stale one gets 409 and the editor keeps the edits. A document save without it gets 422.
- Why: whole-document saves with the last writer winning lost a second tab's edits silently.
- The version counts document saves only: a rename or move never makes an open editor stale.

## Live editing sends id-keyed path ops over one websocket per project

- The client diffs the document into path ops (rules, nodes, edges and subtrees addressed by id, not index) about
  150 ms after typing pauses; the server applies them in arrival order under the project's row lock, logs each
  version in `project_ops` and every client applies every version in order.
- Online, the last writer wins per field; an op inside something another session replaced or deleted is dropped and
  its author told.
- Not CRDT libraries. Automerge orders concurrent writes by (counter, actor), not arrival, inverts delete vs write,
  needs its own binary stored beside the JSON and a new wire format, and was 17 to 64 times slower on 521 rules.
  Yjs/Yrs lets the higher client id win, places reordered items differently, cannot refuse a merge, needs the Y state
  stored too, and was 4 to 11 times slower. Both need glue for every rule above, in Python and JavaScript.
- Behind `DSP_LIVE_COLLAB` (default off); off, the editors save as above.

## One merge core, in Rust, for the server and the browser

- `merge-core/` is built as a PyO3 abi3 extension for the backend and as WebAssembly for the UI; pytest, vitest and
  `cargo test` run the same vectors (`merge-core-bench/vectors/`).
- Why: the two hand-written copies (Python and TypeScript) already disagreed in six places. One implementation gives
  one definition of the semantics; it is not faster or smaller.
- Not the `json-patch` crate: it did about 10 of 439 lines' work and reorders object keys on remove.

## JSON on the wire, not protobuf

- An edit message is 130 to 250 bytes in any format and its codec costs microseconds; the user waits on the debounce
  and the commit. Protobuf made snapshots 23% bigger and 6 to 10 times slower, and turned integers into floats in
  276 of 11,629 messages.
- The one large message is the rule-order op (every rule id); a `move` op would fix it, no format does.

## Several backend nodes share Postgres, nothing else

- Fan-out uses Postgres `LISTEN/NOTIFY` on `project_ops`: the notification is sent on commit, so a listener never
  fetches a row it cannot see, and a listener that reconnects refetches. Not Redis: no new infrastructure.
- Sign-in sessions and OIDC login state are Postgres rows (hashed tokens, update-only renewal). Not the per-process
  cache: a login on one node was a 401 on another, and a sign-out on one node did not reach the others' memory.

## Offline edits merge whole only when nothing overlaps

- Edits made while disconnected go back as one batch, merged only if none of its paths overlaps one changed since its
  base and the merged document brings no new validation error; otherwise it is refused whole and the user keeps
  their version as a download.
- Not an automatic CRDT-style merge: it would accept every combination, including merged states that are invalid
  or that nobody intended, and nobody is told.
- Validation runs on a snapshot without the row lock and commits only if no version came in meanwhile (the third
  attempt holds the lock). Why: validating 521 rules under the lock took 0.35 to 0.5 s and held other editors' round
  trips at 1.5 s under ten batches; now the lock is held about 23 ms (`tree-editor-offline-merge-lock.md`).
- Errors are counted by (type, message), not compared as a set: one more copy of an existing error is new. A merge
  may keep an error either side already had (`max(theirs, mine, theirs + mine - base)` copies); the owner
  confirmed this rule and the three attempts on 2026-10-23.

## The op log is bounded

- Every 100th version stores the whole document; the log keeps a project's latest 5000 versions and 30 days, trimmed
  back to a whole row under the row lock. A rebuild replays at most 100 versions; an editor reconnecting from a
  trimmed version gets a snapshot; an offline batch on one is refused (downloadable).
- 5000 versions and 30 days are the owner's defaults (2026-10-23); `DSP_LIVE_LOG_KEEP_VERSIONS` and
  `DSP_LIVE_LOG_KEEP_DAYS` change them.

## A REST save is logged as a diff

- A REST save logs `diff(stored document, saved document)`, the ops a live editor would send; a save that changes
  nothing still makes a version, with no ops. Live editors apply it like any other version.
- Why: a whole document per save was 68 KB as stored (TOAST compresses it 2.6 times), about 350 MB for 5000 saves of
  521 rules; a one-rule save is now a 115-byte row, 1.1 KB per save with the snapshots
  (`tree-editor-op-log-size.md`). And a root replace overlapped every path: it refused every offline batch based
  before it and dropped every concurrent online edit. Now only edits that overlap what the save changed are refused
  or dropped (the owner's call, 2026-10-23).
- The rows a version is rebuilt from are the snapshots, a new project's first version (whole) and rows logged before
  this change. A save that gives every rule a new id logs 1.4 times the document; the editors keep ids.
- No `lz4` column compression: it stores the documents 8% bigger than the default `pglz`, and a diff row is too small
  to be compressed.

## What it costs

- Every contributor who builds the UI or the backend needs Rust, the wasm32 target, wasm-pack and maturin until CI
  publishes the wheel and `pkg/` (CI only builds and tests them today).
- The image builds `merge-core` in a Rust stage: about 8 minutes on a cold build, 16 s warm.
- The UI cannot sync live without the wasm (+88 KB gzip); if it fails to load, the editors fall back to saves.
- One LISTEN connection and one validator process per backend process; ten simultaneous offline batches on a
  busily edited 521-rule tree finish in about 7 s.
