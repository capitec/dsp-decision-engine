"""Steering a run from outside its code: force a branch's arm or a loop's iteration count, and break on a value.

Forcing works on the condition a branch or loop checks: just after the condition runs, its output is overridden
(recorded as an override, like `set`), so the branch routes, or the loop continues or stops, as asked. Nothing in
the pipeline changes. Value breakpoints pause just after a step that writes the value, when some record newly meets
the condition there.
"""
from __future__ import annotations

import operator

OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def _groups(node, out):
    # Branch or loop path -> its condition's path and output name, and how many arms a branch has.
    kids = node.get("children", ())
    if node["kind"] in ("branch", "loop") and kids and kids[0]["kind"] == "call" and kids[0].get("outputs"):
        out[node["path"]] = {"kind": node["kind"], "cond": kids[0]["path"], "output": kids[0]["outputs"][0],
                             "arms": len(kids) - 1, "max": node.get("maxIterations")}
    for k in kids:
        _groups(k, out)
    return out


def _writers(node, out):
    if node["kind"] == "call":
        for name in node.get("outputs") or ():
            out.setdefault(name, set()).add(node["path"])
    for k in node.get("children", ()):
        _writers(k, out)
    return out


def _under(path, prefix):
    return not prefix or path == prefix or path.startswith(prefix + "/")


class Controls:
    """The forces and value breakpoints of one run, checked at every checkpoint of its session."""

    def __init__(self, ir):
        self.groups = _groups(ir, {})
        self.writers = _writers(ir, {})
        self.forces = []
        self.watches = []
        self.hit = None
        self._matched = {}
        self.session = None

    def set(self, forces=(), watches=()):
        """Replace the forces and watches. A force is `{"path", "arm" | "iterations", "row"?}`; a watch is
        `{"name", "op", "value", "scope"?, "row"?}` or `{"path", "iteration"}` for a loop."""
        for f in forces:
            g = self.groups.get(f["path"])
            if g is None:
                raise KeyError(f"{f['path']!r} is not a branch or loop with a condition step")
            if "arm" in f and not (g["kind"] == "branch" and 0 <= int(f["arm"]) < g["arms"]):
                raise ValueError(f"{f['path']!r} has arms 0 to {g['arms'] - 1}; can't force arm {f['arm']}")
            if "iterations" in f and g["kind"] != "loop":
                raise ValueError(f"{f['path']!r} is not a loop")
        for w in watches:
            if "iteration" in w and self.groups.get(w["path"], {}).get("kind") != "loop":
                raise ValueError(f"{w['path']!r} is not a loop")
            if "op" in w and w["op"] not in OPS:
                raise ValueError(f"unknown comparison {w['op']!r}; use one of {', '.join(OPS)}")
        self.forces, self.watches = list(forces), list(watches)
        self._matched = {}

    def attach(self, session):
        self.session = session
        session.break_at(self.check)
        return self

    def take_hit(self):
        hit, self.hit = self.hit, None
        return hit

    def check(self, cp):
        """Apply any force due at `cp`, then say whether a watch pauses the run here."""
        self.force(cp)
        return self.watch(cp)

    def watch(self, cp):
        """Whether a watch pauses the run at `cp`."""
        return any(self._watch(i, w, cp) for i, w in enumerate(self.watches))

    def force(self, cp):
        """Apply the forces due at `cp`, and say which `(branch or loop, value)` they changed.

        Call it on the checkpoint a run is paused at before moving on.
        """
        done = []
        for f in self.forces:
            g = self.groups[f["path"]]
            if cp.when == "after" and cp.origin.path == g["cond"] and self._force(f, g, cp):
                done.append((f["path"], g["output"]))
        return done

    def _force(self, f, g, cp):
        s = self.session
        current = s.value(g["output"]).to_list()
        is_bool = isinstance(next((v for v in current if v is not None), True), bool)
        if "arm" in f:
            arm = int(f["arm"])
            want = (arm == 0) if is_bool else arm  # a bool condition sends True to arm 0
        else:
            want = (cp.iteration or 1) <= int(f["iterations"])  # checked before iteration k: go on while k <= n
        rows = range(len(current)) if f.get("row") is None else [f["row"]]
        values = list(current)
        for r in rows:
            values[r] = want
        if values != current:
            s.set(g["output"], values)
            return True
        return False

    def _watch(self, i, w, cp):
        if "iteration" in w:
            g = self.groups[w["path"]]
            hit = cp.when == "before" and cp.origin.path == g["cond"] and cp.iteration == int(w["iteration"])
            if hit:
                self.hit = {"watch": i, "text": f"iteration {w['iteration']} of {w['path'].split('/')[-1]}"}
            return hit
        name = w["name"]
        if cp.when != "after" or cp.origin.path not in self.writers.get(name, ()):
            return False
        if not any(_under(cp.origin.path, p) for p in (w.get("scope") or [""])):
            return False
        # What this step wrote, not `session.value`: a loop's carry catches up only after the iteration ends.
        st = self.session.state
        spec = f"{name}@{cp.origin.path}"
        values = st.column(spec).to_list()
        valid = st.valid.get(st.versions(spec)[-1].id)
        rows = [r for r in (range(len(values)) if w.get("row") is None else [w["row"]]) if valid is None or valid[r]]
        test = OPS[w["op"]]

        def meets(v):
            try:
                return v is not None and bool(test(v, w["value"]))
            except TypeError:
                return False

        now = {r for r in rows if meets(values[r])}
        # Only records that newly meet the condition: a value that stays over a limit pauses once, not at every step.
        old = self._matched.get(i, set())
        new = sorted(now - old)
        self._matched[i] = (old - set(rows)) | now  # a record this step didn't write keeps its state
        if new:
            self.hit = {"watch": i, "path": cp.origin.path, "rows": new, "values": [values[r] for r in new],
                        "text": f"{name} {w['op']} {w['value']}"}
        return bool(new)
