# Tree editor

Web editor for decision trees: a FastAPI backend (`backend/`), a React UI (`ui/`), end-to-end tests (`e2e/`) and one image that serves the backend and the built UI.

## Develop

```bash
docker compose up        # UI on :3000, API on :8000, live reload for ui/, backend/ and decikit/
```

Postgres, migrations and `decikit` (editable, from this checkout) are handled by the compose file. Sign up on the login page; the first run installs dependencies, so give it a minute.

Regenerate the UI's API client against the running backend with `./generate-tsx-api-client.sh`.

Every project has a `version` that each save of its document bumps (a rename or move leaves it). The editors send the version they loaded as `base_version` when they save; if someone else saved first, the server answers 409 (`detail.current_version`) and the editor keeps the edits and tells the user to reload. An API caller that omits `base_version` still overwrites.

Every document save is also logged as a version in `project_ops` (the ops that made it) and announced on the Postgres
channel `project_ops`. With `DSP_LIVE_COLLAB=true` the flat-rule editor edits live over `WS
/api/v1/projects/{id}/live`: it pushes each edit about 150 ms after typing pauses, every open editor receives every
version in order, two edits to different fields are both kept, the last writer wins on the same field, and an edit
inside a rule someone else changed or deleted is discarded and its author told. A header badge shows Live or
Offline; Save waits until the server has every edit. READ access receives, WRITE sends. A REST save reaches open
editors as one whole-document replace. The node editor still saves over REST. The log is never trimmed yet.

Edits made while disconnected (a closed socket, or no answer to a ping for about 10 seconds) go back on reconnect as
one offline batch, after the editor has caught up. The server merges it whole only when none of its paths is equal to,
inside or above a path someone else changed since the disconnect (two reorders of the rules conflict only when both
changed the order of the same rules differently) and the merged document has no validation error that neither the
stored document nor the offline editor's own had. Otherwise it refuses the whole batch: the editor shows the stored
document and a banner naming the rules that collided, with Download my version (the offline document, shaped like an
Export). The base document is rebuilt from the log (a new project logs its first version), so a batch older than the
log is refused. A batch is not kept across a page reload.

## Tests

Unit tests:

```bash
cd backend && uv run --with-requirements requirements.txt --with ../../.. --with pytest --with pytest-asyncio pytest
cd ui && pnpm test
```

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
cd ui && pnpm install --frozen-lockfile --filter dsp-re-ui && cd tree-editor/ui && pnpm build:skip-check
cd ../e2e && ./stack.sh up            # Postgres on 55432, backend and UI on http://localhost:8600
./stack.sh test -m smoke              # or -m serve, -m "not serve", test_flat_rule_params.py, -k ...
./stack.sh down                       # stops the backend, removes the container and its volume
```

`E2E_PORT`, `E2E_PG_PORT`, `E2E_RUNTIME=docker` and `E2E_SERVE_PORT` (the first `decikit serve` port, 8650) let
stacks run side by side; temp files and browser profiles go to `E2E_TMPDIR` (`/var/tmp/dtmp-ui-e2e`), not `/tmp`.
The team tests insert their teams with `psql` in the stack's container and skip without it.

`E2E_NODES=2` starts a second backend on `E2E_PORT+1` (same Postgres, logins shared through the `persisted-cache` provider), `DSP_LIVE_COLLAB=true`
turns live editing on, and `./stack.sh restart` restarts the backends but keeps Postgres. `test_live_collab.py` runs only
on such a stack (`E2E_NODES=2 DSP_LIVE_COLLAB=true ./stack.sh up`); the rest of the suite expects live editing off.

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

The image installs `decikit` from the wheel in `dist/`, so build the wheel first and run docker from the repo root:

```bash
uv build
docker build -f ui/tree-editor/Dockerfile -t decider-tree-editor .
docker run -p 8000:8000 decider-tree-editor
```

Releases push `decider-tree-editor:<version>` with the same version as the `decikit` wheel. Settings are read from environment variables and `config.yaml` (see `backend/dsp_decikit/settings.py`).
