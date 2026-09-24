"""Every change to every value, per record, as the run goes: which step made it, in which iteration and arm."""
from __future__ import annotations


def _calls(node, out):
    # Call path -> (inputs, outputs).
    if node["kind"] == "call" and node.get("outputs"):
        out[node["path"]] = (node.get("inputs") or [], node["outputs"])
    for k in node.get("children", ()):
        _calls(k, out)
    return out


class Timeline:
    """What each checkpoint changed. `record` runs at every checkpoint the session passes through."""

    def __init__(self, ir, session):
        self.calls = _calls(ir, {})
        self.session = session
        self.log = []  # (path, when) of every checkpoint recorded, in order
        self.changes = []  # {"at", "path", "iteration", "arm", "name", "rows", "values", "before", "kept", "kept_values"}
        self.undone = []  # what the last rewind took back
        self.inputs = {c: session.frame[c].to_list() for c in session.frame.columns}
        self.now = {k: list(v) for k, v in self.inputs.items()}

    def record(self, cp):
        self.log.append((cp.origin.path, cp.when))
        if cp.when != "after":
            return
        st = self.session.state
        inputs, outputs = self.calls.get(cp.origin.path, ((), ()))
        for name in outputs:
            spec = f"{name}@{cp.origin.path}"
            versions = st.versions(spec)
            if not versions:
                continue
            valid = st.valid.get(versions[-1].id)
            self._note(cp, cp.origin.path, name, st.column(spec).to_list(), valid, kept=True,
                       inputs=[i for i in inputs if i != name])

    def override(self, name):
        """Note a value set by hand at the current checkpoint."""
        cp = self.session.current
        path = "" if cp is None else cp.origin.path
        self._note(cp, f"override@{path}", name, self.session.value(name).to_list(), None)

    def forced(self, group, name):
        """Note a branch's or loop's condition value that a force replaced."""
        self._note(self.session.current, f"force@{group}", name, self.session.value(name).to_list(), None)

    def _note(self, cp, path, name, values, valid, kept=False, inputs=()):
        now = self.now.setdefault(name, [None] * len(values))
        # A row the step didn't write (another arm, a loop that already stopped) keeps its value.
        written = [r for r in range(len(values)) if valid is None or valid[r]]
        rows = [r for r in written if values[r] != now[r]]
        # A step that wrote the value a record already had: "the floor kept 25.2%", for one record's history.
        same = [r for r in written if values[r] == now[r]] if kept else []
        if not rows and not same:
            return
        # A record whose new value is exactly one of the step's inputs: the step passed that value on (a cap that
        # didn't bind), so its history can name where the number was computed.
        via = {}
        for r in rows:
            for i in inputs:
                if i in self.now and self.now[i][r] == values[r] and values[r] is not None:
                    source = self.last(i, r)
                    via[r] = (i, source["path"] if source else None)
                    break
        self.changes.append({"at": len(self.log), "path": path, "iteration": cp and cp.iteration, "arm": cp and cp.arm,
                             "name": name, "rows": rows, "values": [values[r] for r in rows], "before": [now[r] for r in rows],
                             "kept": same, "kept_values": [values[r] for r in same], "via": via})
        for r in rows:
            now[r] = values[r]

    def rewound(self):
        """Forget what happened from the checkpoint the session was rewound to on."""
        cp = self.session.current
        if cp is None:
            return
        # The checkpoint rewound to stays: the session is paused on it and won't pass it again.
        at = self.log.index((cp.origin.path, cp.when)) + 1 if (cp.origin.path, cp.when) in self.log else len(self.log)
        del self.log[at:]
        self.undone = [c for c in self.changes if c["at"] > at]
        self.changes = [c for c in self.changes if c["at"] <= at]
        self.now = {k: list(v) for k, v in self.inputs.items()}
        for c in self.changes:
            now = self.now.setdefault(c["name"], [None] * self.session.frame.height)
            for r, v in zip(c["rows"], c["values"]):
                now[r] = v

    def history(self, name, row=None):
        """The changes to `name`, oldest first; for one record, only the changes to it.

        After a rewind, the changes it undid follow as `pending`: they will happen again as the run goes on.
        """
        out = [e for i, c in enumerate(self.changes) if (e := self._entry(c, i, name, row))]
        # A change undone by a rewind drops out once the run has come back past it.
        out += [dict(e, pending=True, change=-1) for c in self.undone if c["at"] > len(self.log) and (e := self._entry(c, -1, name, row))]
        initial = self.inputs.get(name)
        return {"name": name, "input": initial is not None,
                "initial": None if initial is None else (initial if row is None else initial[row]),
                "changes": out}

    def _entry(self, c, i, name, row):
        if c["name"] != name:
            return None
        e = {k: c[k] for k in ("path", "iteration", "arm")}
        e["change"] = i
        if row is None:
            if not c["rows"]:
                return None
            e.update(rows=len(c["rows"]), values=c["values"][:3], before=c["before"][:3])
        elif row in c["rows"]:
            j = c["rows"].index(row)
            e.update(value=c["values"][j], before=c["before"][j])
            if row in c["via"]:
                e["via"], e["viaPath"] = c["via"][row]
        elif row in c["kept"]:
            v = c["kept_values"][c["kept"].index(row)]
            e.update(value=v, before=v, kept=True)
        else:
            return None
        return e

    def last(self, name, row):
        """The latest change to `name` for record `row`, or None."""
        return next((c for c in reversed(self.changes) if c["name"] == name and row in c["rows"]), None)

    def occurrence(self, change):
        """Which time its step finished, counting from the start: 3 for the third iteration."""
        c = self.changes[change]
        return sum(1 for p in self.log[:c["at"]] if p == (c["path"], "after"))
