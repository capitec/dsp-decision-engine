"""JSON-lines bridge between the VS Code extension and a decider session.

One request per line on stdin, one reply per line on stdout:

    {"id": 1, "cmd": "describe", "file": "app/loan.py"}
    {"id": 1, "ok": true, "result": {...}}

`describe` imports the file and reports its pipelines and IR without running
anything. `start` opens a session; the stepping commands mirror Session.
Replies to stepping commands carry the events produced since the last reply.

Run with `--debugpy PORT` to also listen for a debugpy attach, so the IDE can
drop into the Python of one step.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import select
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import decider_stub as dc  # noqa: E402  (swap for decider.engine.debug when it lands)


def load_module(file):
    sys.path.insert(0, os.path.dirname(os.path.abspath(file)))
    spec = importlib.util.spec_from_file_location("__decider_target__", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assignment_lines(file):
    tree = ast.parse(open(file).read(), file)
    return {t.id: n.lineno for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


def find_pipelines(mod, file):
    """Module-level combinator steps that no other module-level step contains."""
    steps = {k: v for k, v in vars(mod).items() if isinstance(v, dc.Step) and not k.startswith("_")}
    contained = {id(s) for top in steps.values() for _, s in top.walk() if s is not top}
    lines = assignment_lines(file)
    return [{"name": k, "line": lines.get(k), "kind": type(v).__name__}
            for k, v in steps.items() if id(v) not in contained and not isinstance(v, dc.FunctionStep)]


def body_line(fn):
    """First statement of `fn`, where a debugpy breakpoint stops on the call rather than the def."""
    try:
        src, first = inspect.getsourcelines(fn)
        node = ast.parse(textwrap.dedent("".join(src))).body[0]
        return first + node.body[0].lineno - 1
    except (OSError, TypeError, SyntaxError, IndexError, AttributeError):
        return None


def node_json(node):
    o = node.origin
    base = {"path": o.path, "source": o.source, "file": o.file, "line": o.line}
    if isinstance(node, dc.CallNode):
        return {**base, "kind": "call", "inputs": list(node.inputs), "outputs": list(node.outputs),
                "params": node.params, "fills": node.fills, "bodyLine": body_line(node.fn)}
    kind = "branch" if isinstance(node, dc.BranchNode) else "sequence"
    extra = {"modifies": list(node.modifies)} if kind == "branch" else {}
    return {**base, "kind": kind, **extra, "children": [node_json(c) for c in node.children()]}


class Bridge:
    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.mod = None
        self.ir = None
        self.session: dc.Session | None = None
        self.sent_events = 0
        self.pause_requested = False

    def describe(self, file, pipeline=None):
        self.mod = load_module(file)
        pipelines = find_pipelines(self.mod, file)
        if not pipelines:
            raise ValueError(f"{file}: no decider pipeline at module level")
        chosen = pipeline or pipelines[0]["name"]
        self.ir = dc.to_ir(getattr(self.mod, chosen))
        return {"pipelines": pipelines, "pipeline": chosen, "ir": node_json(self.ir),
                "columns": sorted({c for n in dc.call_nodes(self.ir) for c in (*n.inputs, *n.outputs)})}

    def start(self, file, pipeline=None, data=None, params=None, breakpoints=()):
        self.describe(file, pipeline)
        if data is None:
            data = getattr(self.mod, "SAMPLE", None)
        if isinstance(data, str):
            data = json.load(open(data))
        if data is None:
            raise ValueError("no data: pass `data` (rows or a JSON file) or define SAMPLE in the module")
        self.session = dc.Session(self.ir, data, params)
        self.sent_events = 0
        for b in breakpoints:
            self.session.break_at(b)
        return self.status()

    def status(self):
        s = self.session
        events, self.sent_events = s.events[self.sent_events:], len(s.events)
        cur = s.current
        return {"finished": s.finished, "events": events,
                "current": cur and {"path": cur.path, "phase": cur.phase, "depth": cur.depth}}

    def resume(self):
        # Run in step_into slices so a `pause` line arriving on stdin can interrupt.
        s = self.session
        self.pause_requested = False
        while not s.finished:
            s.step_into()
            if s._at_breakpoint():
                s.events[-1]["reason"] = "breakpoint"
                break
            if self.pause_requested or self._pause_pending():
                s.events[-1]["reason"] = "pause"
                break
            s.events.pop()  # the intermediate Paused(step) from step_into
        return self.status()

    def _pause_pending(self):
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except (ValueError, OSError):  # stdin is not a pipe (tests, notebooks)
            return False
        if not ready:
            return False
        line = sys.stdin.readline()
        req = json.loads(line) if line.strip() else {}
        if req.get("cmd") == "pause":
            self._reply(req, {"ok": True})
            return True
        self._reply(req, {"ok": False, "error": "busy: session is running"})
        return False

    def state(self):
        st = self.session.state
        return {"columns": [st.summary(c) for c in sorted(st.columns)]}

    def column(self, name):
        st = self.session.state
        return {"name": name, "values": st.columns[name],
                "versions": [{"producer": p, "values": v} for p, v in st.versions[name]]}

    def lineage(self, name, path=None):
        return dc.lineage(self.ir, name, path)

    def handle(self, req):
        cmd = req["cmd"]
        args = {k: v for k, v in req.items() if k not in ("id", "cmd")}
        if cmd in ("describe", "start", "state", "column", "lineage", "resume"):
            return getattr(self, cmd)(**args)
        s = self.session
        if s is None:
            raise RuntimeError("no session: send start first")
        if cmd == "set":
            s.set(args["name"], args["value"])
        elif cmd in ("step", "step_into", "rewind", "break_at", "clear_break"):
            getattr(s, cmd)(**args)
        elif cmd == "pause":
            return {}
        elif cmd == "events":
            return {"events": s.events}
        else:
            raise ValueError(f"unknown command {cmd!r}")
        return self.status()

    def _reply(self, req, payload):
        self.out.write(json.dumps({"id": req.get("id"), **payload}, default=str) + "\n")
        self.out.flush()

    def serve(self):
        for line in sys.stdin:
            if not line.strip():
                continue
            req = json.loads(line)
            try:
                self._reply(req, {"ok": True, "result": self.handle(req)})
            except Exception as e:  # noqa: BLE001  the IDE shows the message
                self._reply(req, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            if req.get("cmd") == "exit":
                break


def main(argv):
    if "--debugpy" in argv:
        import debugpy
        port = int(argv[argv.index("--debugpy") + 1])
        debugpy.listen(("127.0.0.1", port))
    # Replies on a separate fd keep the protocol clear of print() calls in user steps.
    out = os.fdopen(int(argv[argv.index("--fd") + 1]), "w") if "--fd" in argv else None
    Bridge(out).serve()


if __name__ == "__main__":
    main(sys.argv[1:])
