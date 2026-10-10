# Tree editor

Web editor for decision trees: a FastAPI backend (`backend/`), a React UI (`ui/`), end-to-end tests (`e2e/`) and one image that serves the backend and the built UI.

## Develop

```bash
docker compose up        # UI on :3000, API on :8000, live reload for ui/, backend/ and decikit/
```

Postgres, migrations and `decikit` (editable, from this checkout) are handled by the compose file. Sign up on the login page; the first run installs dependencies, so give it a minute.

Regenerate the UI's API client against the running backend with `./generate-tsx-api-client.sh`.

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
widens the page; two tabs saving one project: the last save wins without a warning; the folder and project move
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
