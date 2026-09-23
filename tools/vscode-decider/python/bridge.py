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
import importlib.util
import inspect
import json
import os
import queue
import sys
import textwrap
import threading
from pathlib import Path

import polars as pl

from decider.engine import Engine
from decider.engine.debug import EVENT
from decider.engine.ir.context import step_map, to_ir
from decider.engine.ir.nodes import BranchNode, CallNode, LoopNode
from decider.steps import FrameStep, FunctionStep, Step

sys.path.insert(0, os.path.dirname(__file__))
from lineage import latest, lineage  # noqa: E402


def load_module(file):
    path = Path(file).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod  # classes defined in it (ConfigurableSteps) resolve by import path
    spec.loader.exec_module(mod)
    return mod


def _assignment_lines(file):
    tree = ast.parse(Path(file).read_text(), file)
    return {t.id: n.lineno for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


def find_pipelines(mod, file):
    """Module-level combinator and config steps that no other module-level step contains."""
    steps = {k: v for k, v in vars(mod).items() if isinstance(v, Step) and not k.startswith("_")}
    contained = {id(s) for top in steps.values() for _, s in top.walk() if s is not top}
    lines = _assignment_lines(file)
    return [{"name": k, "line": lines.get(k), "kind": type(v).__name__} for k, v in steps.items()
            if id(v) not in contained and not isinstance(v, (FunctionStep, FrameStep))]


def _first_statement(fn):
    """First line of `fn`'s body, where a debugpy breakpoint stops on the call rather than the def."""
    try:
        src, first = inspect.getsourcelines(fn)
        node = ast.parse(textwrap.dedent("".join(src))).body[0]
        return first + node.body[0].lineno - 1
    except (OSError, TypeError, SyntaxError, IndexError, AttributeError):
        return None


def _code_location(fn):
    code = getattr(fn, "__code__", None)
    return (code.co_filename, code.co_firstlineno) if code else (None, None)


def node_json(node, steps, located):
    o = node.origin
    step_ = steps.get(o.path)
    file, line = located.get(id(step_), (None, None))
    if isinstance(step_, (FunctionStep, FrameStep)):
        file, line = _code_location(step_.fn)
    base = {"path": o.path, "source": o.source, "file": file, "line": line}
    if isinstance(node, CallNode):
        python = node.reference if node.kind == "row" else node.fn
        rf, rl = _code_location(python)
        return {**base, "kind": "call", "callKind": node.kind,
                "inputs": None if node.inputs is None else [i.name for i in node.inputs],
                "outputs": None if node.outputs is None else [x.name for x in node.outputs],
                "params": {d.name: d.default for d in node.params},
                "python": {"file": rf, "line": rl, "bodyLine": _first_statement(python)} if rf else None}
    kind = "branch" if isinstance(node, BranchNode) else "loop" if isinstance(node, LoopNode) else "sequence"
    extra = {"modifies": list(node.modifies)} if kind == "branch" else {}
    if kind == "loop":
        extra = {"carries": list(node.carries), "maxIterations": node.max_iterations}
    return {**base, "kind": kind, **extra, "children": [node_json(c, steps, located) for c in node.children()]}


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
        self.mod = load_module(file)
        pipelines = find_pipelines(self.mod, file)
        if not pipelines:
            raise ValueError(f"{file}: no decider pipeline at module level")
        self.name = pipeline or pipelines[-1]["name"]
        self.step = getattr(self.mod, self.name)
        self.ir = to_ir(self.step)
        steps = step_map(self.step)
        lines = _assignment_lines(file)
        located = {id(v): (str(Path(file).resolve()), lines[k]) for k, v in vars(self.mod).items()
                   if isinstance(v, Step) and k in lines}
        self.parents = {}
        tree = node_json(self.ir, steps, located)
        self._index(tree, None)
        return {"pipelines": pipelines, "pipeline": self.name, "ir": tree}

    def _index(self, n, parent):
        self.parents[n["path"]] = parent
        for c in n.get("children", ()):
            self._index(c, n["path"])

    def start(self, file, pipeline=None, data=None, params=None, breakpoints=()):
        self.describe(file, pipeline)
        if data is None:
            data = getattr(self.mod, "SAMPLE", None)
        if data is None:
            raise ValueError("no data: pass `data` (rows or a JSON file) or define SAMPLE in the module")
        self.session = Engine().bind(self.step).session(_load_rows(data), params)
        self.sent = 0
        for b in breakpoints:
            self.session.break_at(b)
        return self.status()

    def status(self):
        s = self.session
        events = [EVENT.dump_python(e, mode="json") for e in s.events[self.sent:]]
        self.sent = len(s.events)
        cur = s.current
        return {"finished": s.finished, "events": events,
                "current": cur and {"path": cur.origin.path, "when": cur.when}}

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
            at = None if row is None else self.session.state.column(name, latest(self.session.state, versions, row))[row]
            cols.append({"name": name, "dtype": str(series.dtype), "rows": series.len(),
                         "nulls": series.null_count(), "preview": series.head(5).to_list(), "value": at,
                         "producer": versions[-1].producer or "input", "versions": len(versions)})
        return {"columns": cols, "row": row}

    def column(self, name, row=None):
        st = self.session.state
        versions = [{"producer": v.producer or "input", "values": st.column(name, v).to_list()}
                    for v in self._written(name)]
        if row is not None:
            for v in versions:
                v["values"] = v["values"][row:row + 1]
        return {"name": name, "values": self.session.value(name).to_list(), "versions": versions}

    def handle(self, req):
        cmd = req["cmd"]
        args = {k: v for k, v in req.items() if k not in ("id", "cmd")}
        if cmd in ("describe", "start", "state", "column", "step_out"):
            result = getattr(self, cmd)(**args)
            return self.status() if cmd == "step_out" else result
        s = self.session
        if s is None:
            raise RuntimeError("no session: send start first")
        if cmd == "lineage":
            return lineage(s, **args)
        if cmd in ("step", "step_into", "resume", "rewind", "break_at", "clear_break", "set"):
            try:
                getattr(s, cmd)(**args)
            except Exception as e:  # a step raised: the events hold the Error, the reply the message
                return {**self.status(), "error": f"{type(e).__name__}: {e}"}
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
