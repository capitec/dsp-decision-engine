# Set up a decikit project

Use this to start a project, find your way around one, run or serve it, or get
the decikit MCP server working.

## New project

Inside a uv-managed project (a `pyproject.toml` that depends on decikit):

```bash
uv add "decikit[serve-starlette]"      # decikit and the HTTP server; add [yaml] for flow files
uv run decikit template credit_risk    # writes ./credit_risk
cd credit_risk
uv run pytest -q                       # scores sample_request.json through the handler
uv run decikit build                   # validates configs/<latest>, warms every kernel
uv run decikit serve --workers 1       # POST /invocations, GET /ping
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
```

`decikit build` ends with `built config version 0.0.0 (pipeline
credit_risk.pipeline:build, mode fused)`; the curl answers
`{"income":1000.0,"debt":250.0,"applied_on":"2026-01-15","approved":true}`. A
`FallbackWarning` for a `date` input is normal: that one step runs in Python.

## Layout

The project directory is itself the package, so `import credit_risk` resolves
to it and the directory above goes on `sys.path` (`conftest.py` does this for
pytest).

```
credit_risk/
  pipeline.py            steps, and build() returning the pipeline
  inference.py           Handler(RequestHandler): override input_fn, output_fn, ...
  configs/0.0.0/         one directory per config version
    params.json          the params document
  sample_request.json    one request: tests and warm-up use it
  tests/
  .env                   DECIKIT_* settings
```

- Each argument of `build` receives the config document of that name:
  `def build(limit_band)` gets `configs/<version>/limit_band.json` loaded as a
  step. The template's `build()` takes none.
- `params.json` is keyed flow name, then step, then param:
  `{"credit_risk": {"approved": {"limit": 0.4}}}`. `pipeline.parameters().defaults()`
  prints the whole document with every default.
- A top-level `pipeline.py` outside a package breaks serving; keep
  `<package>/pipeline.py`.

## `.env` and settings

`.env` holds `DECIKIT_API__PIPELINE=credit_risk.pipeline:build`,
`DECIKIT_API__HANDLER=credit_risk.inference:Handler` and
`DECIKIT_CONFIG__BASEPATH=configs`. Run `decikit` commands from the directory
that holds `.env`. Real environment variables win over `.env`. A YAML flow file
serves with `DECIKIT_API__PIPELINE=flow.yaml`. `uv run decikit --help` lists
every command and option; settings are `DECIKIT_*` variables, never invented
flags.

## Config versions

`configs/<version>/` is immutable once a decision has used it. Make a change as
a new version and replay old decisions against the old one.

```python
import json
from decikit.config import JsonFileStore, VersionPart

store = JsonFileStore(basepath="configs")
params = json.loads(json.dumps(store.read("0.0.0").config["params"]))
params["credit_risk"]["approved"]["limit"] = 0.35
store.create_version({"params": params}, bump=VersionPart.MINOR)   # writes configs/0.1.0/params.json
```

`create_version` bumps the latest version and takes the whole config, so pass
every document the version holds (`{"params": ..., "limit_band": ...}`), not only
the changed one. `decikit build 0.1.0` stages a named version; `decikit serve`
serves the latest. The editing tools (configs.md) change a document file in
place: point them at the new version's files, never at a released one.

## The MCP server

The plugin runs `uv run --with "fastmcp>=4.0.10" decikit mcp` as a stdio server
named `decikit`, so it imports the user's pipelines from the user's own
environment.

1. Install, in Claude Code:
   `/plugin marketplace add capitec/dsp-decision-engine`, then
   `/plugin install decikit@decikit`, then restart Claude Code.
2. Check with `/mcp`: `plugin:decikit:decikit` shows `connected`, and the
   tools are named `mcp__plugin_decikit_decikit__<tool>`.
3. Start Claude Code from the project root, the directory with `pyproject.toml`
   (observed on Claude Code 2.1.294: a plugin stdio server starts in the
   directory Claude Code was launched from, so `uv run` finds that project's
   environment and its `decikit`).

Raw per-record tools (`raw_record`, `raw_trace`) answer "raw data disabled"
until the server starts with `--raw`. Opt in only when the user asks, by
registering the server yourself with `--raw` (the `claude mcp add` command
below, with `--raw` appended).

When `/mcp` shows `failed`, or the project is not uv-managed, register the
project's own interpreter instead (verified to connect):

```bash
uv pip install "decikit[mcp]"                   # or: .venv/bin/pip install "decikit[mcp]"
claude mcp add decikit -- "$PWD/.venv/bin/decikit" mcp
```

This names the tools `mcp__decikit__<tool>`. Common causes of `failed`: `uv` is
not on PATH; decikit is not a dependency of the project in the launch
directory (`uv run decikit --help` must work there); Claude Code was started
from a parent directory of several projects.

## VS Code extension

The extension (`ui/vscode`) shows flows as a graph, edits flow files and debugs
a flow. Package it with `pnpm install && pnpm package` there, then
`code --install-extension decikit-vscode-0.1.0.vsix`. It needs `decikit`
importable by the project's Python: set the `decikit.python` setting to
`["uv", "run", "python"]` for a uv project. The editor tools (selection,
reveal, debug) are in [vscode.md](vscode.md).
