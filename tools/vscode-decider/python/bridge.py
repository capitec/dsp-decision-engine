"""JSON-lines bridge between the VS Code extension and a decider debug session.

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

import ast
import json
import os
import queue
import re
import sys
import threading
from pathlib import Path

import polars as pl

from decider.engine import Engine
from decider.engine.debug import EVENT
from decider.engine.debug.edit import swap
from decider.engine.ir.context import step_map, to_ir
from decider.steps import FunctionStep, Step

sys.path.insert(0, os.path.dirname(__file__))
from lineage import latest, lineage  # noqa: E402
from runs import apply_overrides, debug_condition, trace, tree_path  # noqa: E402
from forks import checkpoint_key, merge, sweep  # noqa: E402
from loading import TEXTS, load_module, source_diff  # noqa: E402
from controls import Controls  # noqa: E402
from timeline import Timeline  # noqa: E402
from describing import assignment_lines, find_pipelines, formula, key_column, node_json, values_file  # noqa: E402


def base_params(mod, params):
    """The module's `PARAMS` (its tables and tuned values, as the config store holds them) with `params` on top."""
    return merge(getattr(mod, "PARAMS", None) or {}, params or {}) or None


def _load_rows(data):
    if isinstance(data, str):
        data = json.loads(Path(data).read_text())
    return pl.DataFrame(data)


class Bridge:
    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.lock = threading.Lock()
        self.session = None
        self.sent = 0

    def describe(self, file, pipeline=None):
        self.file, self.mod = file, load_module(file)
        pipelines = find_pipelines(self.mod, file)
        if not pipelines:
            raise ValueError(f"{file}: no decider pipeline at module level")
        self.name = pipeline or pipelines[-1]["name"]
        self.step = getattr(self.mod, self.name)
        self.ir = to_ir(self.step)
        steps = step_map(self.step)
        located = {}
        # Steps assigned in any module of the pipeline's package, the pipeline file first.
        top = self.mod.__name__.split(".")[0]
        mods = [self.mod] + [m for n, m in list(sys.modules.items())
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
                          "values": getattr(self.mod, "PARAMS", None) or {}, "valuesFile": values_file(file)}
        return self.described

    def trace(self, file, pipeline=None, data=None, params=None, overrides=None, row=None, forces=()):
        """Describe and run `file` to the end on `data` (default: its SAMPLE), with `overrides` and `forces` applied."""
        described = self.describe(file, pipeline)
        rows = data if data is not None else getattr(self.mod, "SAMPLE", None)
        if rows is None:
            raise ValueError("no data: pass `data` (rows or a JSON file) or define SAMPLE in the module")
        frame = apply_overrides(_load_rows(rows), overrides, row)
        return {**described, **trace(self.step, frame, base_params(self.mod, params), self.described["ir"], forces), "data": frame.to_dicts(), "key": key_column(frame)}

    def _index(self, n, parent):
        self.parents[n["path"]] = parent
        for c in n.get("children", ()):
            self._index(c, n["path"])

    def start(self, file, pipeline=None, data=None, params=None, breakpoints=(), forces=(), watches=()):
        self.describe(file, pipeline)
        if data is None:
            data = getattr(self.mod, "SAMPLE", None)
        if data is None:
            raise ValueError("no data: pass `data` (rows or a JSON file) or define SAMPLE in the module")
        self.doc = base_params(self.mod, params)
        self.original = self.step
        self.edits = []  # (path, step or None), so each edit can be compared on its own
        self.session = Engine().bind(self.step).session(_load_rows(data), self.doc)
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
            rows = data if data is not None else getattr(self.mod, "SAMPLE", None)
            frame = _load_rows(rows)
            at = None
            params = base_params(self.mod, params)
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
        old, texts = step_map(self.step).get(path), dict(TEXTS)
        new = step_map(getattr(load_module(self.file), self.name)).get(path)
        if new is None:
            raise KeyError(f"{path!r} is no longer in {self.name}")
        self.session.replace(path, new)
        self.step = self.session.executable.step
        self.edits.append((path, new))
        return {"diff": source_diff(old, texts, new, TEXTS), "formula": formula(new.fn) if isinstance(new, FunctionStep) else None}

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
        return {"diff": [], "formula": None}

    def compare_edits(self, path=None):
        """The flow as started and with the edits made since (or only the one at `path`), each run start to end."""
        edited = self.original
        for at, new in self.edits:
            if path is None or at == path:
                edited = swap(edited, at, new)
        frame = self.session.frame
        side = lambda step: {**self.described, **trace(step, frame, self.doc, self.described["ir"], self.controls.forces), "data": frame.to_dicts(), "key": key_column(frame)}  # noqa: E731
        return {"a": side(self.original), "b": side(edited)}

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
        versions = [{"producer": v.producer or "input", "values": st.column(name, v).to_list()}
                    for v in self._written(name)]
        if row is not None:
            for v, version in zip(versions, self._written(name)):
                v["values"] = v["values"][row:row + 1]
                valid = st.valid.get(version.id)
                v["written"] = bool(valid is None or valid[row])
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
            return {**self.status(), "diff": edit.get("diff", []), "formula": edit.get("formula")}
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


def main(argv):
    if "--debugpy" in argv:
        import debugpy
        debugpy.listen(("127.0.0.1", int(argv[argv.index("--debugpy") + 1])))
    # Replies on a separate fd keep the protocol clear of print() calls in user steps.
    out = os.fdopen(int(argv[argv.index("--fd") + 1]), "w") if "--fd" in argv else None
    Bridge(out).serve()


if __name__ == "__main__":
    main(sys.argv[1:])
