"""JSON-lines bridge between an editor (VS Code, JupyterLab) and a decider debug session.

One request per line on stdin, one reply per line on the reply stream:

    {"id": 1, "cmd": "describe", "file": "app/loan.py"}
    {"id": 1, "ok": true, "result": {...}}

`describe` imports the file and reports its pipelines and IR without running
anything. `start` opens a session; the stepping commands mirror `Session`.
Replies to stepping commands carry the events produced since the last reply.
`pause` is answered at once, even while a `resume` is running.

Run with `--debugpy PORT` to also listen for a debugpy attach, so the IDE can
drop into the Python of one step.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from pathlib import Path

import polars as pl

from decider.config import JsonFileStore
from decider.engine import Engine
from decider.engine.debug import EVENT
from decider.engine.debug.edit import swap
from decider.engine.ir.context import step_map, to_ir
from decider.serving.handler import RequestHandler
from decider.serving.parse import coerce_record, has_date
from decider.steps import FunctionStep, Step

from .controls import Controls
from .describing import assignment_lines, find_pipelines, formula, key_column, node_json, values_file
from .forks import checkpoint_key, merge, sweep
from .lineage import latest, lineage
from .loading import load_module
from .runs import apply_overrides, debug_condition, trace, tree_path
from .timeline import Timeline


def base_params(mod, params, default=None):
    """The module's `PARAMS` (its tables and tuned values, as the config store holds them), or `default`
    when it has none, with `params` on top."""
    return merge(getattr(mod, "PARAMS", None) or default or {}, params or {}) or None


def _load_rows(data):
    if data is None:
        raise ValueError("no data: pass `data` (rows or a JSON file) or define SAMPLE in the module")
    if isinstance(data, str):
        data = json.loads(Path(data).read_text())
    if isinstance(data, dict):
        data = [data]  # sample_request.json can be one record, not a list of them
    return pl.DataFrame(data)


def _project_root(file):
    """The nearest ancestor directory of `file` holding a `configs/` directory, as `decider serve` finds one."""
    directory = Path(file).resolve().parent
    for candidate in [directory, *directory.parents]:
        if (candidate / "configs").is_dir():
            return candidate
    return None


def _date_inputs(step):
    return {i.name: i.annotation for i in Engine().bind(step).plan.inputs if has_date(i.annotation)}


def _build(fn, file):
    """Call a serving-style `def build(...)`: its config version's documents, like `decider serve` would.

    Returns the built step, that version's `"params"` document (or `None`), and its
    `sample_request.json` record (or `None`, its dates converted like a request's), from the
    project root the pipeline's file sits in.
    """
    root = _project_root(file) if file else None
    config = {}
    if root is not None:
        store = JsonFileStore(basepath=str(root / "configs"))
        version = store.latest_version()
        config = store.read(version).config if version else {}
    step = RequestHandler(JsonFileStore(), fn).pipeline_fn(config)
    sample = None
    candidate = root and root / "sample_request.json"
    if candidate and candidate.exists():
        sample = coerce_record(json.loads(candidate.read_text()), _date_inputs(step))
    return step, config.get("params"), sample


class Bridge:
    """Answers the editor's requests about one pipeline and its debug session.

    `notebook` returns the module a pipeline is read from when a request names no file: a
    notebook's namespace, with the pipeline, `SAMPLE` rows and `PARAMS` it was handed.
    """

    def __init__(self, out=None, notebook=None):
        self.out = out or sys.stdout
        self.notebook = notebook
        self.lock = threading.Lock()
        self.session = None
        self.sent = 0

    def _load(self, file):
        return load_module(file) if file else self.notebook()

    def describe(self, file=None, pipeline=None):
        self.file, self.mod = file, self._load(file)
        pipelines = (find_pipelines(self.mod, file) if file
                     else [{"name": pipeline, "line": None, "kind": type(getattr(self.mod, pipeline)).__name__}])
        if not pipelines:
            raise ValueError(f"{file}: no decider pipeline: define `def build()` or assign one at module level")
        self.name = pipeline or pipelines[-1]["name"]
        target = getattr(self.mod, self.name)
        if isinstance(target, Step):
            self.step, self._build_params, self._build_sample = target, None, None
        else:
            self.step, self._build_params, self._build_sample = _build(target, file)
        self.ir = to_ir(self.step)
        steps = step_map(self.step)
        located = {}
        # Steps assigned in any module of the pipeline's package, the pipeline file first.
        top = self.mod.__name__.split(".")[0]
        mods = [m for m in [self.mod] if file] + [m for n, m in list(sys.modules.items())
                                                  if (n == top or n.startswith(top + ".")) and m is not self.mod and getattr(m, "__file__", None)]
        for m in mods:
            lines = assignment_lines(m.__file__)
            for k, v in vars(m).items():
                if isinstance(v, Step) and k in lines:
                    located.setdefault(id(v), (str(Path(m.__file__).resolve()), lines[k]))
        self.parents = {}
        tree = node_json(self.ir, steps, located)
        self._index(tree, None)
        params = {path: {k: {kk: vv for kk, vv in info.items() if kk != "used_by" or path == "shared"} for k, info in ps.items()}
                  for path, ps in self.step.parameters().items()}
        # What the flow decides: what it emits, and what its top-level branches set.
        outcome = [n for n in getattr(self.step, "emits", ()) if "@" not in n]
        outcome += [m for c in tree.get("children", ()) if c["kind"] == "branch" for m in c.get("modifies", ())]
        self.described = {"pipelines": pipelines, "pipeline": self.name, "ir": tree, "params": params,
                          "outcome": list(dict.fromkeys(outcome)),
                          "values": getattr(self.mod, "PARAMS", None) or self._build_params or {},
                          "valuesFile": file and values_file(file)}
        return self.described

    def trace(self, file=None, pipeline=None, data=None, params=None, overrides=None, row=None, forces=()):
        """Describe and run `file` to the end on `data` (default: its SAMPLE), with `overrides` and `forces` applied."""
        self.describe(file, pipeline)
        frame = apply_overrides(self._rows(data), overrides, row)
        return self._run(self.step, frame, base_params(self.mod, params, self._build_params), forces)

    def _rows(self, data):
        return _load_rows(data if data is not None else getattr(self.mod, "SAMPLE", None) or self._build_sample)

    def _run(self, step, frame, params, forces):
        return {**self.described, **trace(step, frame, params, self.described["ir"], forces), "data": frame.to_dicts(), "key": key_column(frame)}

    def _index(self, n, parent):
        self.parents[n["path"]] = parent
        for c in n.get("children", ()):
            self._index(c, n["path"])

    def start(self, file=None, pipeline=None, data=None, params=None, breakpoints=(), forces=(), watches=()):
        self.describe(file, pipeline)
        self.doc = base_params(self.mod, params, self._build_params)
        self.original = self.step
        self.edits = []  # (path, step or None), so each edit can be compared on its own
        self.session = Engine().bind(self.step).session(self._rows(data), self.doc)
        self.controls = Controls(self.described["ir"])
        self.controls.set(forces, watches)
        self.controls.session = self.session
        self.timeline = Timeline(self.described["ir"], self.session)
        # First among the breakpoints: `any` stops at the first that pauses, and this one must see every checkpoint.
        self.session.break_at(self._checkpoint)
        self.sent = 0
        # Overrides made so far and where, so forks can replay them; a rewind breaks the replay.
        self.history, self.rewound = [], False
        for b in breakpoints:
            self.session.break_at(b)
        return self.status()

    def status(self):
        s = self.session
        events = [EVENT.dump_python(e, mode="json") for e in s.events[self.sent:]]
        self.sent = len(s.events)
        cur = s.current
        return {"finished": s.finished, "events": events,
                "current": cur and {"path": cur.origin.path, "when": cur.when, **({"iteration": cur.iteration} if cur.iteration else {})},
                "hit": self.controls.take_hit(),
                # What has run since the start or the last rewind, for the graph's ticks.
                "ran": sorted({p for p, w in self.timeline.log if w == "after"})}

    def _checkpoint(self, cp):
        self.timeline.record(cp)
        self._force(cp)
        return self.controls.watch(cp)

    def _set_by_hand(self, entry, row):
        # A forced or overridden value came from you, not from the step that wrote it before.
        c = self.timeline.last(entry["name"], row)
        if c and c["path"].split("@")[0] in ("force", "override"):
            entry.update(setBy=c["path"], was=c["before"][c["rows"].index(row)], inputs=[])
            group = self.controls.groups.get(c["path"].partition("@")[2])
            if group and group["kind"] == "branch":
                entry["arms"] = group["names"]
        for i in entry.get("inputs", ()):
            self._set_by_hand(i, row)

    def _force(self, cp):
        for group, name in self.controls.force(cp):
            self.timeline.forced(group, name)

    def changes(self, name, row=None):
        """Every change to `name` so far (for one record, or all), with the step and iteration that made it."""
        return self.timeline.history(name, row)

    def go_to(self, change=None, path=None, row=None):
        """Go back to just after the step made the timeline's `change`, re-running up to there.

        With `path` instead, go back to just after that step last wrote something (for record `row`).
        """
        if change is None:
            change = next((i for i in range(len(self.timeline.changes) - 1, -1, -1)
                           if self.timeline.changes[i]["path"] == path
                           and (row is None or row in self.timeline.changes[i]["rows"] + self.timeline.changes[i]["kept"])), None)
            if change is None:
                raise ValueError(f"{path} hasn't written anything to go back to")
        c = self.timeline.changes[change]
        if c["path"].split("@")[0] in ("override", "force"):
            raise ValueError("a value you set has no step to go back to")
        n = self.timeline.occurrence(change)
        s = self.session
        # A rewind keeps upstream values as they are now, a loop's carry included: re-run the whole outermost loop.
        loops = [g for g, v in self.controls.groups.items() if v["kind"] == "loop" and c["path"].startswith(g + "/")]
        s.rewind(min(loops, key=len) if loops else c["path"])
        self.rewound = True
        self.timeline.rewound()
        while self.timeline.log.count((c["path"], "after")) < n:
            if s.step_into() is None:
                raise RuntimeError(f"the re-run never reached {c['path']} a {n}th time")
        self.controls.hit = None
        return self.status()

    def rerun(self, path, back=None, when="before"):
        """Go back to just before `path` and run on to `back` (where the run was paused), in one move."""
        s = self.session
        s.rewind(path)
        self.rewound = True
        self.timeline.rewound()
        if back and back != path:
            target = lambda cp: cp.origin.path == back and cp.when == when  # noqa: E731
            s.break_at(target)
            try:
                s.resume()
            finally:
                s.clear_break(target)
        return self.status()

    def set_controls(self, forces=(), watches=()):
        """Replace the run's forces and value breakpoints; they apply from the next checkpoint on."""
        # ponytail: a rewind replays without checking breakpoints, so forces apply only to checkpoints reached
        # by stepping; rewind to the branch or loop itself to re-run it forced.
        self.controls.set(forces, watches)
        return self.status()

    def step_out(self):
        s = self.session
        parent = self.parents.get(s.current.origin.path) if s.current else None
        if parent is None:
            s.resume()
            return
        target = lambda cp: cp.when == "after" and cp.origin.path == parent  # noqa: E731
        s.break_at(target)
        try:
            s.resume()
        finally:
            s.clear_break(target)

    def sweep(self, scenarios, from_here=True, file=None, pipeline=None, data=None, params=None, forces=()):
        """Run every scenario from the paused point (or from the start) next to the unchanged run.

        Each scenario is `{"label", "params", "overrides", "row", "forces"}`. From the start,
        overrides apply to the input rows; from here, to the values at the pause.
        """
        s = self.session
        if from_here and s is not None:
            if self.rewound:
                raise RuntimeError("scenarios can't replay a session that was rewound; restart it first")
            at = checkpoint_key(s)
            result = sweep(self.step, s.frame, s.params, self.history, at, scenarios, self.described["ir"], self.controls.forces)
        else:
            self.describe(file, pipeline)
            frame = self._rows(data)
            at = None
            params = base_params(self.mod, params, self._build_params)
            ir = self.described["ir"]
            base = trace(self.step, frame, params, ir, forces)
            results = []
            for sc in scenarios:
                doc = merge(params, sc.get("params"))
                results.append({"label": sc.get("label"),
                                **trace(self.step, apply_overrides(frame, sc.get("overrides"), sc.get("row")), doc or None,
                                      ir, sc.get("forces") or forces)})
            result = {"baseline": base, "results": results}
        used = s.frame if from_here and s is not None else frame
        return {**result, "describe": self.described, "data": used.to_dicts(), "key": key_column(used),
                "at": None if at is None else {"path": at[0], "when": at[1], "n": at[2]}}

    def skip(self, path):
        """Remove the step at `path` from the paused run and re-run from where it was."""
        self.session.delete(path)
        self.step = self.session.executable.step  # forks replay the edited pipeline
        self.edits.append((path, None))

    def reload_step(self, path):
        """Re-import the pipeline's files and swap the step now at `path` into the paused run."""
        target = getattr(self._load(self.file), self.name)
        new = step_map(target if isinstance(target, Step) else _build(target, self.file)[0]).get(path)
        if new is None:
            raise KeyError(f"{path!r} is no longer in {self.name}")
        self.session.replace(path, new)
        self.step = self.session.executable.step
        self.edits.append((path, new))
        return {"formula": formula(new.fn) if isinstance(new, FunctionStep) else None}

    def restore(self, path):
        """Put the step at `path` back as the run started it, undoing a skip or a code swap, and re-run from there."""
        old = step_map(self.original).get(path)
        if old is None:
            raise KeyError(f"{path!r} is not in the flow as started")
        skipped = any(at == path and new is None for at, new in self.edits)
        rest = [(at, new) for at, new in self.edits if at != path]
        if skipped:
            # The step is gone from the running flow, so its enclosing flow goes back in, with the other edits kept.
            parent = path.rsplit("/", 1)[0] if "/" in path else self.original.name
            rebuilt = self.original
            for at, new in rest:
                rebuilt = swap(rebuilt, at, new)
            self.session.replace(parent, step_map(rebuilt)[parent])
        else:
            self.session.replace(path, old)
        self.step = self.session.executable.step
        self.edits = rest

    def compare_edits(self, path=None):
        """The flow as started and with the edits made since (or only the one at `path`), each run start to end."""
        edited = self.original
        for at, new in self.edits:
            if path is None or at == path:
                edited = swap(edited, at, new)
        run = lambda step: self._run(step, self.session.frame, self.doc, self.controls.forces)  # noqa: E731
        return {"a": run(self.original), "b": run(edited)}

    def _names(self):
        plan = self.session.executable.plan
        return sorted({v.name for v in plan.versions} | set(self.session.state.chains))

    def _written(self, name):
        st = self.session.state
        inputs = [v for v in self.session.executable.plan.versions if v.producer is None and v.name == name]
        return inputs + [v for v in st.chains.get(name, ()) if v.id in st.values]

    def state(self, row=None):
        cols = []
        for name in self._names():
            try:
                series = self.session.value(name)
            except KeyError:
                continue  # not produced yet
            versions = self._written(name)
            # Mid-iteration, a loop's carry still holds the last iteration's value; the timeline has what was written last.
            now = self.timeline.now.get(name)
            if now is None:
                now = series.to_list()
                if row is not None:
                    now[row] = self.session.state.column(name, latest(self.session.state, versions, row))[row]
            at = None if row is None else now[row]
            cols.append({"name": name, "dtype": str(series.dtype), "rows": series.len(),
                         "nulls": now.count(None), "preview": now[:5], "value": at,
                         "producer": versions[-1].producer or "input", "versions": len(versions)})
        return {"columns": cols, "row": row, "key": key_column(self.session.frame)}

    def column(self, name, row=None):
        st = self.session.state
        rows = slice(None) if row is None else slice(row, row + 1)
        versions = [{"producer": v.producer or "input", "values": st.column(name, v).to_list()[rows]} for v in self._written(name)]
        return {"name": name, "values": self.session.value(name).to_list(), "versions": versions}

    def handle(self, req):
        cmd = req["cmd"]
        args = {k: v for k, v in req.items() if k not in ("id", "cmd")}
        if cmd in ("describe", "start", "state", "column", "step_out", "trace", "sweep", "compare_edits", "set_controls", "changes", "go_to", "rerun"):
            result = getattr(self, cmd)(**args)
            return self.status() if cmd == "step_out" else result
        s = self.session
        if s is None:
            raise RuntimeError("no session: send start first")
        if cmd == "lineage":
            row = args.get("row")
            now = self.timeline.now.get(args["name"])
            # Not set yet for this record (after going back past it): nothing to break down, and not an error.
            if row is not None and args["name"] not in s.frame.columns and (now is None or now[row] is None):
                return {"name": args["name"], "producer": None, "value": None, "inputs": [], "unset": True}
            entry = lineage(s, **args)
            if args.get("row") is not None:
                self._set_by_hand(entry, args["row"])
            return entry
        if cmd == "tree_path":
            return tree_path(s, **args)
        if cmd == "debug_condition":
            return {"condition": debug_condition(s, **args)}
        if cmd in ("skip", "reload_step", "restore"):
            edit = getattr(self, cmd)(**args) or {}  # a WiringError changes nothing and comes back as the reply's error
            self.timeline.rewound()
            return {**self.status(), "formula": edit.get("formula")}
        if cmd in ("step", "step_into", "resume", "rewind", "break_at", "clear_break", "set"):
            if cmd in ("step", "step_into", "resume") and s.current is not None:
                self._force(s.current)  # paused on a condition, its breakpoint check already passed
            try:
                getattr(s, cmd)(**args)
            except Exception as e:  # a step raised: the events hold the Error, the reply the message
                return {**self.status(), "error": f"{type(e).__name__}: {e}"}
            if cmd == "set":
                self.history.append((checkpoint_key(s), args["name"], args["value"]))
                self.timeline.override(args["name"])
            if cmd == "rewind":
                self.timeline.rewound()
            self.rewound |= cmd == "rewind"
            return self.status()
        if cmd == "exit":
            return {}
        raise ValueError(f"unknown command {cmd!r}")

    def _reply(self, req, payload):
        with self.lock:
            self.out.write(json.dumps({"id": req.get("id"), **payload}, default=str) + "\n")
            self.out.flush()

    def serve(self, stdin=sys.stdin):
        # A reader thread, so `pause` lands while the main thread is inside `resume`.
        requests: queue.Queue = queue.Queue()

        def read():
            for line in stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if req.get("cmd") == "pause":
                    if self.session is not None:
                        self.session.pause()
                    self._reply(req, {"ok": True, "result": {}})
                else:
                    requests.put(req)
            requests.put(None)

        threading.Thread(target=read, daemon=True).start()
        while (req := requests.get()) is not None:
            try:
                self._reply(req, {"ok": True, "result": self.handle(req)})
            except Exception as e:  # noqa: BLE001  the IDE shows the message
                self._reply(req, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            if req.get("cmd") == "exit":
                break


def main(argv=sys.argv[1:]):
    if "--debugpy" in argv:
        import debugpy
        debugpy.listen(("127.0.0.1", int(argv[argv.index("--debugpy") + 1])))
    # Replies on a separate fd keep the protocol clear of print() calls in user steps.
    out = os.fdopen(int(argv[argv.index("--fd") + 1]), "w") if "--fd" in argv else None
    Bridge(out).serve()

