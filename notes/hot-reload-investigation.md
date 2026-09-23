# Hot reload for live sessions (spike)

Question: can an open debug `Session` pick up an edited step by itself, (a) when
a notebook cell is re-run and (b) when `pipeline.py` is edited on disk?
Answer: yes, with one diff-and-rebuild primitive (`session.reload`) plus two
thin ways of getting the new pipeline. About 110 lines of logic, dev-time only.

## What the prototype does

- **`Session.reload(pipeline)`** (`engine/debug/edit.py`). Resolves the new
  pipeline, then lists every checkpoint of both plans in order as
  `(when, path, content)` (`engine/debug/hot.py: keys`). Content of a call
  is `fingerprint(fn)` (and of `reference`), inputs, outputs, params *with
  their defaults* (`ParamDecl ==` leaves `default` out), consts, `nogil`
  and the `(name, producer)` it reads. A `ConfigurableStep` is keyed by class
  name, `model_dump_json()`, relabels and the fingerprints of its class's
  methods, not by its IR, because a tree's IR holds array addresses that
  change on every load. Composite nodes key on their own fields (emits,
  modifies, carries), not their children. The first index where the two
  lists differ is where the run restarts: everything before it is the same
  code reading the same versions, so its values are kept.
- **Applying** reuses T6.5's machinery, split out of `_edit` into
  `_apply(root, plan, target)`: carry values over by `(name, producer)`,
  `skip` upstream nodes, replay to the target, pause with reason `"edit"`.
  `replace`/`delete` are now two small callers of it. One `Edited` event per
  changed path; `Edited.action` gains `"add"`. Nothing changed → no events,
  nothing re-runs. A pipeline that doesn't wire raises and changes nothing.
- **Snapshot at open.** Keys are taken when the session opens and after every
  edit. Otherwise a notebook that redefines a *helper* a step calls would
  compare the old step against the new globals and see no change.
- **Notebook: `session.watch(source)`** registers an IPython `post_run_cell`
  hook that calls `reload(source())`. `watch(lambda: pipeline)` picks up a
  re-run pipeline cell. `watch(lambda: flow(income, ratio))` also picks up a
  cell that only redefines `ratio` or a helper, because the lambda looks the
  names up again each time: re-resolving by name done by Python's own scoping.
  A failed reload emits `Error("reload failed: ...", None)` and re-raises
  (IPython prints it). The session keeps its pipeline.
- **Files: `ModuleWatcher("pkg.pipeline:pipeline")`** (same `module:attr`
  contract as `DECIDER_API__PIPELINE`; a function is called with no
  arguments). `poll()` stats the files of every imported module under the
  target's directory, excluding `site-packages`, `decider` and `__main__`. On a
  change it drops *all* of them from `sys.modules` and imports the target
  again. It does not `importlib.reload` the changed file alone, which would
  leave `from features import ratio` in other modules pointing at the old
  function; a fresh import runs them in dependency order. It deletes their
  `.pyc` first, because a `.pyc` stamps whole seconds and a same-size edit
  within one second would load old code. A failed import puts the old modules
  back and raises. `watch("pkg.pipeline:pipeline")` polls on every cell run,
  the `%autoreload` model.
- Tests: `tests/debug/test_session_reload.py` (10). Covers replacing in every
  mode, an identical rebuild being a no-op, insert → `add` and delete, a
  changed param default, a bad reload, the IPython `InteractiveShell` flow
  (helper redefinition, broken cell), and editing an imported module on disk:
  no stale reference, only the edited function compiles, and a syntax error
  keeps the old modules. `ipython` is added to the dev group. Full suite:
  1482 passed.

## Measurements

28-core Linux box, numba 0.67, CPython 3.14. Each figure is the time from the
edit to the updated `output()`: poll + re-import + diff + replay + resume. `Nc`
is the number of numba compilations.

| | interpreted | stepped | fused |
|---|---|---|---|
| file, 3-step flow, edit a constant in step 2 | 5–6 ms / 0c | 186 ms / 2c | 142–151 ms / 1c |
| file, same, back to an earlier version | – | 7 ms / 0c | 7 ms / 0c |
| file, IR.md §7 example, edit `cap_private` default | 19–28 ms / 0c | 26–39 ms / 0c | 29–34 ms / 0c |
| file, comment-only edit, 3-step / §7 | 5 / 21 ms | 5 / 28 ms | 5 / 39 ms |
| cell, 3-step: `reload` vs per-path `replace` | 1.5 vs 0.9 ms | 182 vs 189 ms | 148 vs 156 ms |
| cell, insert a step at the end: `reload` | 1.0 ms | 169 ms | 153 ms |

- `keys()` for the §7 pipeline takes 1.6 ms, paid on every session open and
  every edit. A no-op `reload` of §7 takes 1.6 ms.
- The compiles are the edited step's dispatcher plus the fused glue of the one
  kernel that contains it. Stepped wraps a call in a kernel too, hence 2.
  Unchanged steps in a re-imported module compile nothing: the dispatcher
  cache is keyed by fingerprint, so a new function object with old content
  hits. Neither does going back to a version seen before. A param-default
  edit never compiles (params are arguments).
- **§7 always re-runs from `join_bureau`**, even for a comment-only edit.
  Its fingerprint covers the global `BUREAU`, a DataFrame, which is keyed by
  identity. Re-importing makes a new one, so the step counts as changed and
  everything after it re-runs. That is correct (the frame might have
  changed) and cheap here (21–39 ms), but noisy: one `Edited` per reload.
  Hashing frames or arrays by value in `fingerprint` would fix it if it
  matters.

**Per-path edits vs rebuild: rebuild.** Their cost is the same (compile-bound
in compiled modes; ~1 ms apart interpreted). N changed paths as N `replace`
calls means N re-resolves and N replays. `swap` can't insert, and can't
replace an anonymous root flow at all (`replace("", ...)` raises `KeyError`),
so a top-level insert would need a new `add` primitive. The first-difference
rebuild handles modify, insert, delete, reorder and relabel with one code path,
and restarts at the same point a sequence of per-path edits would.

## Re-resolving functions by `__qualname__` in `__main__`: not sound

Swapping a pipeline's `fn` for whatever `__main__.<qualname>` holds now would
spare the user re-running the flow cell. Cases that break it:

- `ratio_v1 = ratio` followed by a new `def ratio`: the step the user kept on
  purpose gets swapped.
- `<lambda>` and `make_rule.<locals>.rule` have no global to find.
- `@step(output=...)` globals are `FunctionStep`s. Swapping only `fn` ignores
  a changed `output=`, and swapping the whole step drops the pipeline's
  `.named()`, `.bind()` and relabels.
- A `ConfigurableStep` instance would have to be re-validated through the new
  class.

Patching `__code__` in place (`%autoreload`'s trick) mutates objects that the
IR cache and frozen steps assume never change. `watch(lambda: flow(...))` gets
the same convenience from ordinary name lookup, with none of these cases.
`Origin.source` doesn't help either: in a notebook it is `__main__:ratio` for
every generation of `ratio`.

## Risks

- **Module-level state re-runs** on every file change: data loads, caches,
  connections. Any module under the watched root is re-imported, even one
  that didn't change (by design, to avoid stale references). Objects created
  from the old modules still reference the old classes; the registry's
  same-import-path replacement keeps `ConfigurableStep.load` pointing at the
  new class.
- **Identity-keyed globals** (frames, arrays, dicts) make a step count as
  changed on every reload. The cost is extra re-runs, never stale values.
- **Wiring noise:** inserting a producer before C changes what C reads, so C
  is reported `replace` too. That is accurate but chatty.
- **Threads:** `Session` isn't thread-safe. A watcher thread must not call
  `reload` while a command runs; it has to go through whatever serialises
  commands (see below).
- **Partial edits:** saving a half-written file gives a `SyntaxError` or a
  `WiringError`. Both leave the old pipeline and are reported once per save,
  not once per poll.
- **`Error` event reuse:** today `Error` means "the run stopped, only
  `rewind` works". A reload error doesn't stop the run. The UI must tell the
  two apart (`path is None` and the `"reload failed"` prefix), or
  `Edited`/`Error` needs a separate `ReloadFailed` event. Recommend the
  latter if this ships.
- `ModuleWatcher`'s root is one directory. Code imported from a sibling
  directory isn't watched. Add a `paths=` argument when someone needs it.

## Websocket adapter

`Edited` per change plus the existing `Paused("edit")` is enough for the
event stream. Two things are missing:

1. The reload must run on the adapter's command queue, the same worker that
   runs `resume`. A ~5-line poll task (`await asyncio.sleep(0.5)`; if
   `watcher.poll()` returns a pipeline, enqueue a Python-only
   `Reload(pipeline)` command, like `Replace`) does it.
2. After an `add` or `delete` the UI's step tree is stale, and there is no
   command that returns the pipeline's paths. Add a `structure` request
   (paths and kinds from `step_map`), or put the path list in `Edited`.

Breakpoints on deleted paths stay registered and never fire, which is harmless.

## Serving: never hot reload

Serving has explicit `stage`/`activate`/`rollback` for config documents. A
code change is a new deploy: the decision audit has to map to one code
version. An in-process reload would also compile on the request path
(142–186 ms per edited step, against a 60 µs p99 budget) and swap modules
under in-flight requests. For development servers, a process-restarting
`uvicorn --reload` is the right tool. `ModuleWatcher` stays in `engine/debug`
and nothing in `serving/` imports it except the session websocket.

## Recommendation

- **Primitive:** `Session.reload(pipeline)`, which diffs checkpoint
  sequences by content, applies them in one rebuild and emits `Edited` per
  path. It lives in `engine/debug/edit.py`, and `replace`/`delete` become
  thin callers of the shared `_apply`.
- **Notebook:** `session.watch(lambda: flow(...))`, or
  `watch(lambda: pipeline)`, through an IPython `post_run_cell` hook. No
  `__main__` qualname re-resolution.
- **Files:** `ModuleWatcher("module:attr")` with a fresh re-import of every
  module under the target's directory. It is polled by
  `session.watch("module:attr")` in notebooks, and by a queue-fed poll task
  in `session_ws` for the UI.
- **Before shipping:** a `ReloadFailed` event instead of reusing `Error`, the
  websocket poll task and a `structure` request, and hashing frames and
  arrays by value in `fingerprint` so data globals stop forcing re-runs.
- **Serving:** no hot reload.
