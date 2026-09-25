# The flow debugger in JupyterLab

`tools/jupyterlab-decider` is a JupyterLab 4 prebuilt extension that hosts the shared UI
(`@decider/ui`, see `notes/decider-ui.md`) next to notebooks and flow files.

## Quickstart

```sh
cd tools && pnpm install                        # links @decider/ui, builds it
cd jupyterlab-decider
uv sync                                         # decider, decider-bridge, ipykernel; jupyterlab (dev)
pnpm build                                      # tsc, then `jupyter labextension build` into decider_jupyter/labextension
uv run jupyter labextension develop . --overwrite
uv run jupyter lab
```

- **A flow file:** right-click a `.py` file in the file browser, *Debug decider flow*. It runs on
  the module's `SAMPLE` rows and `PARAMS`, like VS Code's *Run flow* with SAMPLE.
- **A notebook:** `from decider_jupyter import debug; debug(pipeline, df)` opens the panel on a
  pipeline defined in the notebook, run on `df` (polars, pandas or a list of records), with an
  optional `params` document. Re-running the cell that defines a step, then *Use edited code*,
  swaps it into the paused run.

The panel's toolbar has the run controls: Start (again to restart), Continue, Step over,
Step in, Step out, Stop. Everything else is the UI VS Code shows.

A wheel (`uv build`) carries the built frontend as shared data, so `pip install` of it is
enough for a user; build the frontend first.

## Transport: a kernel comm

The kernel runs the bridge (`decider_bridge.Bridge`) and the panel talks to it over a comm
named `decider`: `{id, cmd, ...args}` in, `{id, ok, result | error}` out, the same requests
VS Code's debug adapter sends down the bridge's stdin.

Chosen over a server extension because live notebook objects are why you'd debug in Jupyter:
a pipeline built in a cell, a DataFrame loaded in another. A server extension would run the
bridge in the server's process or a child, where those objects don't exist; it could only
debug files, as VS Code does. A comm also needs no server-side code at all, so the Python
package is kernel-side only.

- **Notebooks:** `debug()` opens the comm from the kernel; the frontend registers the target
  on every notebook's kernel and opens a panel when it fires. The bridge reads the pipeline
  from a snapshot of the notebook's namespace taken on each describe, so a re-run cell counts.
- **Files:** the panel starts a kernel session in the file's folder (the kernel's working
  directory is then the file's, as when you run it), imports `decider_jupyter` (which
  registers the comm target), and opens the comm itself. Closing the panel shuts that kernel.
- **Comparisons and scenarios** (`trace`, a sweep from the start) run on a fresh `Bridge` in
  the same kernel (`fresh: true`), so the debug session stays where it was. VS Code runs them
  in separate processes; in-process is simpler and fast enough, but a flow in a package is
  re-imported for them, which the paused session doesn't notice because it holds its own
  step objects.

Limits of a comm: the kernel answers one message at a time, so there is no *pause* while a
long `resume` runs (VS Code's bridge reads `pause` on a thread), and a notebook cell that is
running blocks the panel of that notebook's pipeline until it finishes.

## Where the bridge lives

The bridge moved from `tools/vscode-decider/python` to `tools/decider-bridge/decider_bridge`,
a package with its own `pyproject.toml`; its tests are `tools/decider-bridge/tests`. VS Code
runs it as `python -m decider_bridge` with `tools/decider-bridge` on `PYTHONPATH`; the
Jupyter package depends on it by path. `Bridge(notebook=...)` is the only change for Jupyter:
with no file in a request, the pipeline, `SAMPLE` and `PARAMS` come from the module that
callable returns. A packaged `.vsix` would need `decider-bridge` copied in; nothing packages
one today.

`readEvents` (folding session events into what has run, tree visits and edits) moved from
the VS Code adapter into `@decider/ui`'s model, since both hosts fold the same events.

## The host

`src/host.ts` is the Jupyter counterpart of VS Code's `adapter.ts` plus the message handling
in `extension.ts`: it keeps where the run is, turns each bridge status into the UI's `status`
and `state` messages, and answers the UI's messages. `can` is `reveal`, `runTo`, `step`: Open
source opens the file in a JupyterLab editor at the step's line (files under the server root
only; a notebook's own functions have none), and Run to / Run through drive the session
directly. Left out: Step into the Python (debugpy), view diff, compare with a git revision,
maximise.

A finished run ends like a VS Code debug session: the graph's ticks and the state clear.

## Theme

`style/base.css` maps `--jp-*` onto every `--decider-*` variable without a UI fallback, on
`.jp-DeciderPanel`, plus link and code styles VS Code's webviews give for free. Light and dark
themes both read well (`test/e2e/shots/`).

## Fixes the UI needed

- **Newer Chromium returns promises from `scrollIntoView` and `scrollTo`.** Two effects written
  `useEffect(() => el?.scrollTo(0, 0))` handed React that promise as their cleanup, and the
  next re-render threw "destroy is not a function", unmounting the whole view. VS Code's
  Electron is older, so only the Jupyter host showed it. Effects now use block bodies.
- **A value's breakdown outlived its record.** Stopping a run (or leaving a focused record)
  while a breakdown showed rendered its per-record history as a batch one and crashed. The UI
  now drops the breakdown when the focused record changes, and the host drops an answer that
  arrives after the run stopped.

## Tests

```sh
uv run pytest -q                                # tools/jupyterlab-decider: the comm glue, a notebook pipeline
pnpm e2e                                        # Playwright on a real JupyterLab, screenshots in test/e2e/shots
```

The e2e harness (`test/e2e/lab.ts`) starts `jupyter lab` on a free port with a token, serving
a fresh copy of `loan.py`, with `--expose-app-in-browser` so tests can run commands, and
drives Playwright's Chromium headless. A failed test leaves a `failed-*.png`.
