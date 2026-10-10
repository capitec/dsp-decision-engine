"""Work out the `module:qualname` entry of a live flow."""
from __future__ import annotations

import sys

from decikit.ir import iter_nodes, to_ir


def infer_entry(step) -> str:
    """The `module:name` of the module-level variable that is `step`, or `""`.

    Looks in the module of the flow's first user step; never calls anything.
    """
    nodes = [n.origin for n in iter_nodes(to_ir(step))]
    module = next((o.source.partition(":")[0] for o in nodes if not o.source.startswith("decikit")), "")
    mod = sys.modules.get(module)
    if mod is None or module == "__main__":
        return ""
    return next((f"{module}:{name}" for name, obj in vars(mod).items() if obj is step), "")
