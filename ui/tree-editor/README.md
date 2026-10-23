# Tree editor

Web editor for decision trees: a FastAPI backend (`backend/`), a React UI (`ui/`), end-to-end tests (`e2e/`) and one image that serves the backend and the built UI.

## Develop

```bash
docker compose up        # UI on :3000, API on :8000, live reload for ui/, backend/ and decikit/
```

Postgres, migrations and `decikit` (editable, from this checkout) are handled by the compose file. Sign up on the login page; the first run installs dependencies and builds `merge-core` (below), so give it a few minutes. After editing `merge-core/`, rerun `docker compose run --rm merge-core` and restart the backend.

Regenerate the UI's API client against the running backend with `./generate-tsx-api-client.sh`.

Every project has a `version` that each save of its document bumps (a rename or move leaves it). `PUT /api/v1/projects/{id}` with `project_data` must also send `base_version`, the version the caller loaded (`GET` returns it as `version`):

- without it the server answers 422 and names the field;
- if someone else saved first, it answers 409 (`detail.code` is `stale_version`, `detail.current_version` the version now stored), and the editor keeps the edits, shows "Save Refused" and asks the user to reload;
- an update of only `name`, `description` or `folder_id` needs no `base_version` and ignores one.

Every document save is also logged as a version in `project_ops` (the ops that made it) and announced on the Postgres
channel `project_ops`. With `DSP_LIVE_COLLAB=true` both editors edit live over `WS
/api/v1/projects/{id}/live`: it pushes each edit about 150 ms after typing pauses, every open editor receives every
version in order, two edits to different fields are both kept, the last writer wins on the same field, and an edit
inside a rule or node someone else changed or deleted is discarded and its author told. A header badge shows Live or
Offline; Save waits until the server has every edit. READ access receives, WRITE sends. A REST save reaches open
editors as one whole-document replace. The node editor's nodes, edges (key `id`, else `source:target`) and subtrees
are addressed by id; node positions are part of the document and sync, selection and measured sizes do not.
Opening a tree sends nothing until an edit (the editor's own form of a stored document goes with the first edit), and
a version from someone else that moves a node turns Auto-arrange off in this browser, so it does not lay their move
out again. Two browsers connecting nodes at once can make a cycle; the validation errors show it. A
browser that cannot load the merge-core wasm still opens both editors, says live editing is unavailable there, and
saves with Save over REST (a 409 when someone saved since). Every 100th version's log row also holds the whole
document, and the log keeps a project's latest 5000 versions and none older than 30 days
(`DSP_LIVE_LOG_KEEP_VERSIONS`, `DSP_LIVE_LOG_KEEP_DAYS`), trimmed back to a row holding a whole document, so any kept
version is rebuilt by replaying at most 100 versions. An editor reconnecting from a version the log no longer keeps
gets the whole document.

Edits made while disconnected (a closed socket, or no answer to a ping for about 10 seconds) go back on reconnect as
one offline batch, after the editor has caught up. The server merges it whole only when none of its paths is equal
to, inside or above a path someone else changed since the disconnect (unless both left the same value there; two
reorders of the rules conflict only when both changed the order of the same rules differently) and the merged
document has no validation error beyond those the stored document or the offline editor's own had (counted by type
and message: one more copy of an existing error is new; the same error added by each side on its own is not).
Otherwise it refuses the whole batch: the editor shows the stored document and a banner naming the rules (or nodes)
that collided, with Download my version (the offline document, shaped like an Export). The merge is validated without
holding the project's row lock and committed only if no other version came in meanwhile; the third attempt holds the
lock throughout. A validator that does not answer within 30 s is replaced by a new process, and a batch it was
checking is refused (still downloadable). The base document is rebuilt from the log (a new project logs its first
version, and the migration that adds snapshots logs the current version of every project older than that), so a batch
made on a version the log no longer keeps is refused and stays downloadable. A message is logged at most once per
client: a resend is not applied again, and a batch names the earlier messages it contains so one applied while the
editor reconnected is skipped, and one delivered late after the batch is not applied again. A batch is not kept
across a page reload.

The socket checks the session and access on every message and every 15 s. A live edit near the session's expiry
renews it by the same rule as an HTTP request (a ping does not, so an idle open tab still expires), and the server
sends the new status cookie for the page to set. Signing out (on any node), an expired session or removed access
closes the socket with 1008 and a reason, also at the handshake; the editor stops reconnecting, shows "Not live" and a
toast, and keeps unsent edits in the editor (Export saves them); signing in again through the session dialog
reconnects and sends them as an offline batch. A message Postgres or the merge core would refuse
every time (NUL, a lone surrogate, a non-finite number, nesting past 64 levels) is rejected up front. A close the
server reports as its failure on what the editor sent (1009 over 4 MiB, 1011 a server error, 1007 not JSON), right
after the editor sent its edits and with none answered, counts against them: after three in a row the editor gives
them up and shows the offline banner with Download my version. Any other drop (1013 for a database outage, a lost
connection, a missed ping, a restart) is never counted, so an outage never hands edits back. The fan-out sends to a
project's sockets at once, each with a 5 s limit; a socket that stops reading is dropped, closed with 1013 and
catches up when it reconnects.

## merge-core

The live-editing ops (diff, apply, conflict checks) are written once, in Rust, in `merge-core/`: the backend runs it as
a Python extension (`dsp_decikit.services.live_ops` re-exports it) and the browser as WebAssembly
(`ui/src/lib/live-ops.ts` wraps it). `merge-core-bench/` holds the vectors every build runs, the frozen reference
implementations and the measurements behind this choice.

Nothing built from it is committed or published, so every contributor who runs the backend or the UI needs:

- Rust 1.95 or newer through rustup, and the wasm target: `rustup target add wasm32-unknown-unknown`;
- wasm-pack: `cargo install --locked wasm-pack@0.15.0`;
- a C linker (`cc`, e.g. `gcc`), which maturin needs to link the extension;
- maturin needs no install: uv fetches it when it builds `--with ../merge-core` (and rebuilds when a Rust file
  changes). zig is needed only for the portable wheel the image and CI build (`uvx --from 'maturin[zig]==1.15.0'
  maturin build --release --zig --compatibility manylinux_2_28`, run in `merge-core/`; uv fetches zig too).

Then `pnpm build:wasm` (in `ui/`) writes `merge-core/pkg/`: run it before `pnpm test`, `pnpm build:skip-check`,
`pnpm dev` or `tsc`, and again after editing the Rust. Without Rust each step stops with a message naming what to
install (`build:wasm`, the vite and vitest configs, and the backend's import of `merge_core`); `tsc` alone reports
only the missing module. `cargo test` in `merge-core/` runs the vectors in Rust;
`.github/workflows/tree-editor-merge-core.yml` runs them against the wheel and the wasm on every pull request that
changes the crate.

## Tests

Unit tests:

```bash
cd backend && uv run --with-requirements requirements.txt --with ../../.. --with ../merge-core --with pytest --with pytest-asyncio pytest
cd ui && pnpm build:wasm && pnpm test
cd merge-core && cargo test
```

`backend/tests/test_sessions_across_nodes.py` starts two backend processes on one Postgres and runs only with
`TREE_EDITOR_DB=1`: bring up the e2e stack (below) and add `TREE_EDITOR_DB=1 DSP_DB__PORT=55432` to the backend command.

End to end (`e2e/`: Python, Playwright and pytest in headless Chromium) covers the user stories of both editors and
the platform around them, every real tree in the repo through import, save, reload and export, each export run in
`decikit`, and a handful served by a real `decikit serve`. Each test signs up a fresh user and seeds its projects
through the API, then drives the behaviour under test in the UI.

| area | files |
|---|---|
| parameters in flat rules (the reported bug and its variants) | `test_flat_rule_params.py`, `test_flat_rule_cases.py` |
| flat-rule editor | `test_flat_rules_{conditions,structure,rule_list,validation,editor_behaviour,outputs,types_params}.py` |
| node (v3) editor | `test_node_editor_{structure,conditions,cases_composite,outputs_params,canvas,validation}.py` |
| every real tree | `test_real_trees.py` (round trip, canonical compare), `test_real_trees_run.py` (the export runs like its source), `test_real_tree_imports.py` (the Import dialog, step documents, files that are no tree) |
| a real `decikit serve` | `test_serve_exported.py` |
| platform | `test_projects_folders.py`, `test_folder_tree.py`, `test_sharing_permissions.py`, `test_teams.py`, `test_sessions.py`, `test_resilience.py`, `test_import_export.py`, `test_auth.py`, `test_tree_editor.py` |

Page objects: `tree_ui.py`, `flat_rules_ui.py`, `node_ui.py`, `real_trees_ui.py`, `platform_ui.py`; API seeding:
`seed.py`; documents: `flat_docs.py` and `real_trees.py`, the tree inventory, which `e2e/data/README.md` explains:
each tree, what is excluded and why, what the canonical comparison may ignore, and how the serve tests build their
project. The pure logic (parameter references, validation-error parsers, the flat-rule store's save) has vitest tests
under `ui/src/**/__tests__/`.

Markers: `smoke` (19 core stories, about 70 seconds), `full` (everything else, set automatically), `serve` (starts
real `decikit` servers), `slow` (large trees). A failing test leaves a screenshot and a trace in
`<repo>/.scratch/ui-e2e-artifacts/`.

With podman, `e2e/stack.sh` runs Postgres in a memory-capped container and the backend, which serves the built UI,
on one port; no compose file is needed:

```bash
cd ui && pnpm install --frozen-lockfile --filter dsp-re-ui && cd tree-editor/ui && pnpm build:wasm && pnpm build:skip-check
cd ../e2e && ./stack.sh up            # Postgres on 55432, backend and UI on http://localhost:8600
./stack.sh test -m smoke              # or -m serve, -m "not serve", test_flat_rule_params.py, -k ...
./stack.sh down                       # stops the backend, removes the container and its volume
```

`E2E_PORT`, `E2E_PG_PORT`, `E2E_RUNTIME=docker` and `E2E_SERVE_PORT` (the first `decikit serve` port, 8650) let
stacks run side by side; temp files and browser profiles go to `E2E_TMPDIR` (`/var/tmp/dtmp-ui-e2e`), not `/tmp`.
The team tests insert their teams with `psql` in the stack's container and skip without it.

`E2E_NODES=2` starts a second backend on `E2E_PORT+1` (same Postgres, so sessions are shared),
`DSP_LIVE_COLLAB=true` turns live editing on, and `./stack.sh restart` restarts the backends but keeps Postgres.
`test_live_collab.py`, `test_live_offline.py` and `test_live_node_editor.py` run only on such a stack
(`E2E_NODES=2 DSP_LIVE_COLLAB=true ./stack.sh up`); `test_live_ops_unavailable.py` runs either way. The smoke set
and `test_sessions.py` pass with live editing on as well as off: a live Save waits for "Project Saved" instead of the
REST reply, and a test of a REST-only story is skipped when live, naming the live test that covers it. The rest of
the suite is run with live editing off.

With docker compose, `docker compose --profile e2e run --rm e2e` runs the whole suite against the dev stack;
`-e PYTEST_ADDOPTS="-m smoke"` picks a subset. The suite was run under podman with `stack.sh`; the compose route is
kept from before and was not rerun. Against a stack you already have running (needs `uv run playwright install
chromium` once): `cd e2e && E2E_BASE_URL=http://localhost:3000 uv run --with-requirements requirements.txt --with
../../.. pytest`.

Known failures are strict `xfail`s whose reason says what is wrong. A fix turns one into a failure, so the marker
goes in the same commit as the fix. Open at the time of writing: the node editor's browser Back loses unsaved
edits; inf, nan and 1e400 thresholds are stored as 0 or null in both editors; renaming or deleting an input field a
rule reads is not reported; deleting a custom type a column uses gives only a generic confirm; unnamed all-mode
flat rules are renamed `Rule N`, which renames their output columns; the node editor draws a refused (cyclic)
connection, ignores a stray node silently, binds only Backspace (not Delete), has a dead Add Subtree button, a Clear
Workspace dialog that promises an undo it lacks and a canvas 80px taller than the page; a 150-character rule name
widens the page; the folder and project move
dialogs, Create Subfolder's parent, folder-create and team-create messages, Delete all contents (it moves subfolders
to the root), folder shares not listing the folder's projects and subfolders and the refresh popup closed without a
reason. The 521-rule sets are
`xfail(run=False)` in the editor round trips: the flat-rule editor renders every rule and exceeds 3 GB.

## Image

The image installs `decikit` from the wheel in `dist/`, so build the wheel first and run docker from the repo root. It builds `merge-core` itself, in a Rust stage (the extension wheel for the runtime, the wasm for the UI stage):

```bash
uv build
docker build -f ui/tree-editor/Dockerfile -t decider-tree-editor .
docker run -p 8000:8000 decider-tree-editor
```

Releases push `decider-tree-editor:<version>` with the same version as the `decikit` wheel. Settings are read from environment variables and `config.yaml` (see `backend/dsp_decikit/settings.py`).

### More than one backend node

Any number of backend processes can share one Postgres behind a load balancer, without sticky routing. Sign-in
sessions and OIDC login state live in Postgres (table `cache_entries`, keyed by a SHA-256 of the token, never the token
itself), so a sign-in, renewal or sign-out on one node holds on every other node from its next request, and an OIDC
login can return to a different node than the one it started on. Each OIDC login state works once and expires after
`login_session_timeout_s`. Expired rows are removed whenever a session or an OIDC login starts.

- Give every node the same settings: database, `auth` (cookie names, session durations, `api_key.salt_value`, OIDC
  providers and `redirect_url_override`). Expiry is checked against each node's clock, so keep the clocks in sync.
- The `cache` setting (`DSP_CACHE`) only caches data-source schemas and previews, never sessions. With `in-memory`, the
  default, each node keeps its own copy, so a cached preview can differ between nodes.
- Upgrading from a version that kept sessions in `cache` signs everyone out once.
