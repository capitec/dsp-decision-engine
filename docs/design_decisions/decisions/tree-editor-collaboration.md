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

## What it costs

- Every contributor who builds the UI or the backend needs Rust, the wasm32 target, wasm-pack and maturin until CI
  publishes the wheel and `pkg/` (CI only builds and tests them today).
- The image builds `merge-core` in a Rust stage: about 8 minutes on a cold build, 16 s warm.
- The UI cannot sync live without the wasm (+88 KB gzip); if it fails to load, the editors fall back to saves.
- One LISTEN connection and one validator process per backend process.
