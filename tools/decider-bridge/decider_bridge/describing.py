"""What the flow panel shows about a pipeline before it runs: its pipelines, and its IR as JSON with source locations."""
from __future__ import annotations

import ast
import hashlib
import inspect
import re
import textwrap
from pathlib import Path

from decider.engine.ir.nodes import BranchNode, CallNode, LoopNode
from decider.steps import FrameStep, FunctionStep, Step


def assignment_lines(file):
    tree = ast.parse(Path(file).read_text(), file)
    return {t.id: n.lineno for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


def _build_line(file):
    """The line of a top-level `def build` in `file`, the serving default entry point (`pipeline:build`)."""
    tree = ast.parse(Path(file).read_text(), file)
    return next((n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"), None)


def find_pipelines(mod, file):
    """Combinator and config steps assigned at the top of `file` that no other one there contains,
    plus a top-level `def build` there, if any."""
    lines = assignment_lines(file)  # imported sub-flows aren't this file's pipelines
    steps = {k: v for k, v in vars(mod).items() if isinstance(v, Step) and not k.startswith("_") and k in lines}
    contained = {id(s) for top in steps.values() for _, s in top.walk() if s is not top}
    pipelines = [{"name": k, "line": lines.get(k), "kind": type(v).__name__} for k, v in steps.items()
                 if id(v) not in contained and not isinstance(v, (FunctionStep, FrameStep))]
    build_line = _build_line(file)
    if build_line is not None and inspect.isfunction(getattr(mod, "build", None)):
        pipelines.append({"name": "build", "line": build_line, "kind": "build"})
    return pipelines


def _statement_line(fn, index):
    # The first statement is where a debugpy breakpoint stops on the call rather than the def.
    try:
        src, first = inspect.getsourcelines(fn)
        node = ast.parse(textwrap.dedent("".join(src))).body[0]
        return first + node.body[index].lineno - 1
    except (OSError, TypeError, SyntaxError, IndexError, AttributeError):
        return None


def _short_source(fn, limit=15):
    """A step function's source when it is short enough to read in the details pane."""
    try:
        lines = textwrap.dedent(inspect.getsource(fn)).rstrip().splitlines()
    except (OSError, TypeError):
        return None
    return "\n".join(lines) if len(lines) <= limit else None


def values_file(file):
    """The file a module's `PARAMS = ...` reads, when that line names one: `json.loads(... / "params.json" ...)`."""
    for line in Path(file).read_text().splitlines():
        if line.startswith("PARAMS") and (m := re.search(r"[\"']([\w./-]+\.(?:json|ya?ml|toml))[\"']", line)):
            return m.group(1)
    return None


def formula(fn):
    """The expression a one-line step returns, e.g. `min(pl_raw_rate, repo_rate + cap_margin)`."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError, IndexError):
        return None
    body = [b for b in getattr(node, "body", ()) if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    return ast.unparse(body[0].value) if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value else None


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
                "formula": formula(python) if isinstance(step_, FunctionStep) else None,
                "body": _short_source(python) if isinstance(step_, FunctionStep) else None,
                "table": step_.expression.model_dump(mode="json") if hasattr(step_, "expression") and hasattr(step_, "rows") else None,
                "python": {"file": rf, "line": rl, "bodyLine": _statement_line(python, 0), "endLine": _statement_line(python, -1)} if rf else None}
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
