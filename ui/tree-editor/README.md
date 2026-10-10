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

End to end (drives a real browser through the main user stories, and runs the tree it exports with `decikit`):

```bash
docker compose --profile e2e run --rm e2e
```

Against a stack you already have running (needs `uv run playwright install chromium` once):

```bash
cd e2e && E2E_BASE_URL=http://localhost:3000 uv run --with-requirements requirements.txt --with ../../.. pytest
```

## Image

The image installs `decikit` from the wheel in `dist/`, so build the wheel first and run docker from the repo root:

```bash
uv build
docker build -f ui/tree-editor/Dockerfile -t decider-tree-editor .
docker run -p 8000:8000 decider-tree-editor
```

Releases push `decider-tree-editor:<version>` with the same version as the `decikit` wheel. Settings are read from environment variables and `config.yaml` (see `backend/dsp_decikit/settings.py`).
