"""Converting a paused session's changes into an experiment draft.

Every structured change (a `set`, a forced branch or loop) becomes a declared
override; every code edit and console mutation is dropped and listed, never
silently omitted.
"""


def draft_changes(history, forces, edits):
    """`history` (`(checkpoint, name, value)` overrides), `forces` and `edits` as a draft:
    `{"converted": [declared overrides], "dropped": [unconvertible changes]}`.
    """
    converted = []
    for key, name, value in history:
        path = key[0] if key else ""
        converted.append({"target": f"{name}@{path}" if path else name, "value": value, "source": "set"})
    for f in forces:
        converted.append({"target": f"force@{f['path']}", "value": f.get("arm", f.get("iterations")),
                          "source": "force", "row": f.get("row")})
    dropped = [{"kind": "code_edit", "path": path,
                "detail": "step skipped" if new is None else "step code replaced"} for path, new in edits]
    dropped.append({"kind": "console_mutation",
                    "detail": "Python console mutations are not tracked; any such changes are not in this draft"})
    return {"converted": converted, "dropped": dropped}
