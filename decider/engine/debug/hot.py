from __future__ import annotations

import contextlib
import dataclasses
import importlib
import importlib.util
import os
import pkgutil
import sys
import types
from typing import Any

from pydantic import BaseModel

from decider.engine.compile.fingerprint import fingerprint
from decider.engine.ir.nodes import CallNode
from decider.engine.wiring.plan import Call, Plan

# Fields holding a node's children, which get their own checkpoints and keys.
_CHILDREN = ("origin", "children_", "condition", "arms", "body")


def keys(plan: Plan, steps: dict) -> list[tuple[str, str, Any]]:
    """Every checkpoint of `plan` in order, as `(when, path, content)`; equal prefixes compute equal values."""
    from decider.engine.debug.edit import _order

    order, spans = _order(plan.root)
    by_path = {r.node.origin.path: r for r in spans}
    own = {p: _key(r, steps.get(p)) for p, r in by_path.items()}
    return [(when, path, own[path]) for when, path in order]


def same(a: Any, b: Any) -> bool:
    # A value without a plain `==` (an array, a frame) counts as changed.
    try:
        return bool(a == b)
    except Exception:
        return False


def _key(r: Any, step: Any) -> Any:
    node = r.node
    reads = isinstance(r, Call) and r.reads is not None and tuple((v.name, v.producer) for v in r.reads)
    if isinstance(step, BaseModel):
        # A config's IR holds addresses rebuilt on every load, so key it by its data and its class's code.
        cls = type(step)
        code = tuple(fingerprint(f) for c in cls.__mro__[:cls.__mro__.index(BaseModel)]
                     for f in vars(c).values() if isinstance(f, types.FunctionType))
        return cls.__qualname__, step.model_dump_json(), step.reads, step.writes, code, reads
    if isinstance(node, CallNode):
        return (node.kind, fingerprint(node.fn), node.reference and fingerprint(node.reference), node.inputs,
                node.outputs, tuple((p, p.default) for p in node.params), node.consts, node.nogil, reads)
    return type(node).__name__, *(getattr(node, f.name) for f in dataclasses.fields(node) if f.name not in _CHILDREN)


class ModuleWatcher:
    """Re-imports the user's code behind `"module:attr"` when one of its files changes.

    Watches every imported module whose file is under `root` (default: the
    target module's directory), outside `site-packages` and `decider`. On a
    change all of them are dropped from `sys.modules` and the target imported
    afresh, so a `from features import ratio` elsewhere never keeps the old
    function. A failed import puts the old modules back.

    In a notebook, `session.watch("credit.pipeline:pipeline")` polls one
    after every cell; the debug websocket takes `watch=` and polls one for
    its client. In a script, poll it yourself.

    Example::

        watcher = ModuleWatcher("credit.pipeline:pipeline")
        s = watcher.get().session(df)
        s.resume()
        # ... edit credit/features.py and save ...
        if (new := watcher.poll()) is not None:
            s.reload(new)       # re-runs from the first changed step
            s.resume()
    """

    def __init__(self, target: str, root: str | None = None):
        self.target = target
        module = importlib.import_module(target.partition(":")[0])
        self.root = os.path.abspath(root or os.path.dirname(module.__file__))
        self.stamps = self._stamps()

    def get(self) -> Any:
        """The target as imported now; a function is called for the pipeline it builds."""
        value = pkgutil.resolve_name(self.target)
        return value if not callable(value) or hasattr(value, "to_ir") else value()

    def poll(self) -> Any:
        """The re-imported pipeline if a watched file changed since the last poll, else `None`."""
        stamps = self._stamps()
        if stamps == self.stamps:
            return None
        # Reported once, not on every poll until it's fixed.
        self.stamps = stamps
        old = self._modules()
        for name, module in old.items():
            del sys.modules[name]
            # A .pyc stamps whole seconds; a same-size edit within one would load the old code.
            with contextlib.suppress(OSError):
                os.remove(importlib.util.cache_from_source(module.__file__))
        importlib.invalidate_caches()
        try:
            new = self.get()
        except BaseException:
            for name in self._modules():
                del sys.modules[name]
            sys.modules.update(old)
            raise
        self.stamps = self._stamps()
        return new

    def _modules(self) -> dict[str, types.ModuleType]:
        return {n: m for n, m in list(sys.modules.items())
                if (f := getattr(m, "__file__", None)) and os.path.abspath(f).startswith(self.root + os.sep)
                and "site-packages" not in f and n != "__main__" and n.partition(".")[0] != "decider"}

    def _stamps(self) -> dict[str, tuple[int, int] | None]:
        out = {}
        for name, m in self._modules().items():
            try:
                st = os.stat(m.__file__)
                out[name] = (st.st_mtime_ns, st.st_size)
            except OSError:
                out[name] = None
        return out

