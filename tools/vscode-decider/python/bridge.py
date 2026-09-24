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
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import queue
import re
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import polars as pl

from decider.engine import Engine
from decider.engine.debug import EVENT
from decider.engine.debug.edit import swap
from decider.engine.ir.context import step_map, to_ir
from decider.engine.ir.nodes import BranchNode, CallNode, LoopNode
from decider.steps import FrameStep, FunctionStep, Step

sys.path.insert(0, os.path.dirname(__file__))
from lineage import latest, lineage  # noqa: E402
from runs import apply_overrides, debug_condition, trace, tree_path  # noqa: E402
from forks import checkpoint_key, merge, sweep  # noqa: E402


# Each loaded module file's text as it was loaded, so an edit can be diffed against the code that ran.
_TEXTS: dict[str, str] = {}


def load_module(file):
    """Import `file` from its source as it is now: by dotted name when it sits in a package, so its imports resolve."""
    # Cached bytecode is keyed on the file's mtime in whole seconds and its size, so an edit saved within the
    # same second as the last load, at the same length, would run the old code. Compile into a fresh cache.
    with tempfile.TemporaryDirectory(prefix="decider-pyc-") as fresh:
        before, sys.pycache_prefix = sys.pycache_prefix, fresh
        try:
            mod = _import(file)
        finally:
            sys.pycache_prefix = before
    top = mod.__name__.split(".")[0]
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and (m.__name__ == top or m.__name__.startswith(top + ".")):
            _TEXTS[f] = Path(f).read_text()
    return mod


def _import(file):
    path = Path(file).resolve()
    root, parts = path.parent, [path.stem]
    while (root / "__init__.py").exists():
        parts.insert(0, root.name)
        root = root.parent
    sys.path.insert(0, str(root))
    name = ".".join(parts)
    if len(parts) > 1:
        for mod_name in [m for m in sys.modules if m == parts[0] or m.startswith(parts[0] + ".")]:
            del sys.modules[mod_name]  # a fresh import, so edits since the last describe count
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # classes defined in it (ConfigurableSteps) resolve by import path
    spec.loader.exec_module(mod)
    return mod


def base_params(mod, params):
    """The module's `PARAMS` (its tables and tuned values, as the config store holds them) with `params` on top."""
    return merge(getattr(mod, "PARAMS", None) or {}, params or {}) or None


def _assignment_lines(file):
    tree = ast.parse(Path(file).read_text(), file)
    return {t.id: n.lineno for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


def find_pipelines(mod, file):
    """Combinator and config steps assigned at the top of `file` that no other one there contains."""
    lines = _assignment_lines(file)  # imported sub-flows aren't this file's pipelines
    steps = {k: v for k, v in vars(mod).items() if isinstance(v, Step) and not k.startswith("_") and k in lines}
    contained = {id(s) for top in steps.values() for _, s in top.walk() if s is not top}
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


def _values_file(file):
    """The file a module's `PARAMS = ...` reads, when that line names one: `json.loads(... / "params.json" ...)`."""
    for line in Path(file).read_text().splitlines():
        if line.startswith("PARAMS") and (m := re.search(r"[\"']([\w./-]+\.(?:json|ya?ml|toml))[\"']", line)):
            return m.group(1)
    return None


def _formula(fn):
    """The expression a one-line step returns, e.g. `min(pl_raw_rate, repo_rate + cap_margin)`."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError, IndexError):
        return None
    body = [b for b in getattr(node, "body", ()) if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    return ast.unparse(body[0].value) if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value else None


def _source_diff(old, old_texts, new, new_texts):
    """The lines that differ between two steps' Python, each read from its file as it was loaded, as `-`/`+` lines."""
    def lines(step_, texts):
        code = getattr(getattr(step_, "fn", None), "__code__", None)
        text = code and texts.get(code.co_filename)
        if not text:
            return []
        return [line.rstrip("\n") for line in inspect.getblock(text.splitlines(True)[code.co_firstlineno - 1:])]
    return [line for line in difflib.unified_diff(lines(old, old_texts), lines(new, new_texts), lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]


def _code_location(fn):
    code = getattr(fn, "__code__", None)
    return (code.co_filename, code.co_firstlineno) if code else (None, None)


def _fingerprint(fn, step_):
    """Changes when the code or config behind a node changes; compared across revisions."""
    try:
        text = inspect.getsource(fn)
    except (OSError, TypeError):
        text = repr(fn)
    if hasattr(step_, "model_dump_json"):
        text += step_.model_dump_json()
    return hashlib.sha1(text.encode()).hexdigest()[:12]


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
                "params": {d.name: d.default for d in node.params}, "code": _fingerprint(python, step_),
                "doc": (inspect.getdoc(python) or "").split("\n")[0],
                "formula": _formula(python) if isinstance(step_, FunctionStep) else None,
                "table": step_.expression.model_dump(mode="json") if hasattr(step_, "expression") and hasattr(step_, "rows") else None,
                "python": {"file": rf, "line": rl, "bodyLine": _first_statement(python)} if rf else None}
    kind = "branch" if isinstance(node, BranchNode) else "loop" if isinstance(node, LoopNode) else "sequence"
    extra = {"modifies": list(node.modifies)} if kind == "branch" else {}
    if kind == "loop":
        extra = {"carries": list(node.carries), "maxIterations": node.max_iterations}
    return {**base, "kind": kind, **extra, "children": [node_json(c, steps, located) for c in node.children()]}


def key_column(frame):
    """The column that names a record (`client_id`, `id`), so views can say "client_id 2" not "row 1"."""
    for c in frame.columns:
        if c == "id" or c.endswith("_id") or c.startswith("id_"):
            return {"name": c, "values": frame[c].to_list()}
    return None


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
            lines = _assignment_lines(m.__file__)
            for k, v in vars(m).items():
                if isinstance(v, Step) and k in lines:
                    located.setdefault(id(v), (str(Path(m.__file__).resolve()), lines[k]))
        self.parents = {}
        tree = node_json(self.ir, steps, located)
        self._index(tree, None)
        params = {path: {k: {kk: vv for kk, vv in info.items() if kk != "used_by" or path == "shared"} for k, info in ps.items()}
                  for path, ps in self.step.parameters().items()}
        self.described = {"pipelines": pipelines, "pipeline": self.name, "ir": tree, "params": params,
                          "values": getattr(self.mod, "PARAMS", None) or {}, "valuesFile": _values_file(file)}
        return self.described

    def trace(self, file, pipeline=None, data=None, params=None, overrides=None, row=None):
        """Describe and run `file` to the end on `data` (default: its SAMPLE), with `overrides` applied."""
        described = self.describe(file, pipeline)
        rows = data if data is not None else getattr(self.mod, "SAMPLE", None)
        if rows is None:
            raise ValueError("no data: pass `data` (rows or a JSON file) or define SAMPLE in the module")
        frame = apply_overrides(_load_rows(rows), overrides, row)
        return {**described, **trace(self.step, frame, base_params(self.mod, params)), "data": frame.to_dicts(), "key": key_column(frame)}

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
        self.doc = base_params(self.mod, params)
        self.original = self.step
        self.edits = []  # (path, step or None), so each edit can be compared on its own
        self.session = Engine().bind(self.step).session(_load_rows(data), self.doc)
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

    def sweep(self, scenarios, from_here=True, file=None, pipeline=None, data=None, params=None):
        """Run every scenario from the paused point (or from the start) next to the unchanged run.

        Each scenario is `{"label", "params", "overrides", "row"}`. From the start,
        overrides apply to the input rows; from here, to the values at the pause.
        """
        s = self.session
        if from_here and s is not None:
            if self.rewound:
                raise RuntimeError("scenarios can't replay a session that was rewound; restart it first")
            at = checkpoint_key(s)
            result = sweep(self.step, s.frame, s.params, self.history, at, scenarios)
        else:
            self.describe(file, pipeline)
            rows = data if data is not None else getattr(self.mod, "SAMPLE", None)
            frame = _load_rows(rows)
            at = None
            params = base_params(self.mod, params)
            base = trace(self.step, frame, params)
            results = []
            for sc in scenarios:
                doc = merge(params, sc.get("params"))
                results.append({"label": sc.get("label"),
                                **trace(self.step, apply_overrides(frame, sc.get("overrides"), sc.get("row")), doc or None)})
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
        old, texts = step_map(self.step).get(path), dict(_TEXTS)
        new = step_map(getattr(load_module(self.file), self.name)).get(path)
        if new is None:
            raise KeyError(f"{path!r} is no longer in {self.name}")
        self.session.replace(path, new)
        self.step = self.session.executable.step
        self.edits.append((path, new))
        return {"diff": _source_diff(old, texts, new, _TEXTS), "formula": _formula(new.fn) if isinstance(new, FunctionStep) else None}

    def compare_edits(self, path=None):
        """The flow as started and with the edits made since (or only the one at `path`), each run start to end."""
        edited = self.original
        for at, new in self.edits:
            if path is None or at == path:
                edited = swap(edited, at, new)
        frame = self.session.frame
        side = lambda step: {**self.described, **trace(step, frame, self.doc), "data": frame.to_dicts(), "key": key_column(frame)}  # noqa: E731
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
            at = None if row is None else self.session.state.column(name, latest(self.session.state, versions, row))[row]
            cols.append({"name": name, "dtype": str(series.dtype), "rows": series.len(),
                         "nulls": series.null_count(), "preview": series.head(5).to_list(), "value": at,
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
        if cmd in ("describe", "start", "state", "column", "step_out", "trace", "sweep", "compare_edits"):
            result = getattr(self, cmd)(**args)
            return self.status() if cmd == "step_out" else result
        s = self.session
        if s is None:
            raise RuntimeError("no session: send start first")
        if cmd == "lineage":
            return lineage(s, **args)
        if cmd == "tree_path":
            return tree_path(s, **args)
        if cmd == "debug_condition":
            return {"condition": debug_condition(s, **args)}
        if cmd in ("skip", "reload_step"):
            edit = getattr(self, cmd)(**args) or {}  # a WiringError changes nothing and comes back as the reply's error
            return {**self.status(), "diff": edit.get("diff", []), "formula": edit.get("formula")}
        if cmd in ("step", "step_into", "resume", "rewind", "break_at", "clear_break", "set"):
            try:
                getattr(s, cmd)(**args)
            except Exception as e:  # a step raised: the events hold the Error, the reply the message
                return {**self.status(), "error": f"{type(e).__name__}: {e}"}
            if cmd == "set":
                self.history.append((checkpoint_key(s), args["name"], args["value"]))
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
