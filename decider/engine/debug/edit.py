from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable

from decider.engine.debug.events import Edited, Error, Paused
from decider.engine.debug.hot import ModuleWatcher, keys, same
from decider.engine.ir.nodes import CallNode, iter_nodes
from decider.engine.wiring import resolve
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, Sequence, Version
from decider.exceptions import WiringError

if TYPE_CHECKING:
    from decider.engine.run.runners.base import Checkpoint
    from decider.steps.base import Step

# Fields through which a step holds its members.
_MEMBERS = ("steps", "arms", "condition", "body")


class Edits:
    # The half of `Session` that edits the pipeline mid-run.

    def replace(self, path: str, step: Any) -> Checkpoint | None:
        """Put `step` (a step or a function) in place of the step at `path`, then re-run from there.

        The new step takes the old one's name, so it sits at the same path
        and its params, breakpoints and `name@path` lookups carry over. Any
        step can be replaced: a function, a flow, a branch arm, a config.

        Only what the change affects runs again: values upstream of `path`,
        overrides included, are kept as they are, and the session pauses just
        before the new step (reason `"edit"`), or where it was paused if that
        comes first. Once the run was over, `resume()` finishes it again.
        Raises (and changes nothing) if the edited pipeline doesn't wire,
        e.g. a later step reads a name only the old step produced.

        Example::

            s = pipeline.session(df)
            s.resume()
            s.replace("term/cap_by_income", step(cap_by_income_v2))
            s.resume()
            s.output()["term_cap"]      # computed by cap_by_income_v2
        """
        from decider.steps.base import as_step

        return self._edit("replace", path, as_step(step))

    def delete(self, path: str) -> Checkpoint | None:
        """Remove the step at `path` from its flow, then re-run from where it was.

        Like `replace`: upstream values are kept and the session pauses at
        the step that followed it. Deleting one layer of a waterfall leaves
        readers with the layer before it; deleting the only producer of a
        name something later reads raises a `WiringError` and changes nothing.

        Example::

            s.delete("term/cap_by_income")
            s.resume()
        """
        return self._edit("delete", path, None)

    def reload(self, pipeline: Any) -> Checkpoint | None:
        """Switch to `pipeline`, an edited version of the session's one, re-running from its first change.

        Steps are compared by content (function code, config data, params,
        wiring), not by identity, so re-running the cells or re-importing the
        module that build the pipeline is enough. Values computed before the
        first change are kept, overrides included; the session pauses there
        (reason `"edit"`) or where it was paused if that comes first. One
        `Edited` event per changed path (`"replace"`, `"add"` or `"delete"`);
        nothing happens if nothing changed. Raises, and changes nothing, if
        `pipeline` doesn't wire.

        Example::

            s.resume()
            s.reload(flow(disposable_income, ratio_v2, affordable))
            s.resume()
        """
        from decider.engine import step_map
        from decider.steps.base import as_step

        root = as_step(pipeline)
        plan = resolve(root)
        old, new = self._keys, keys(plan, step_map(root))
        first = next((i for i, (a, b) in enumerate(zip(old, new)) if a[:2] != b[:2] or not same(a[2], b[2])),
                     None if len(old) == len(new) else min(len(old), len(new)))
        if first is None:
            return self.current
        was = {p: k for w, p, k in old if w == "before"}
        now = {p: k for w, p, k in new if w == "before"}
        edits = [Edited("delete", p) for p in was if p not in now]
        edits += [Edited("add" if p not in was else "replace", p) for p in now if p not in was or not same(was[p], now[p])]
        return self._apply(root, plan, lambda pos: pos(*new[first][:2]) if first < len(new) else None, edits,
                           "the pipeline", new)

    def watch(self, source: Any) -> Callable[[], Any]:
        """Reload after every notebook cell: from `source()`, or from `"module:attr"` when its files change.

        `lambda: pipeline` picks up a re-run pipeline cell; a function that
        builds the pipeline (`lambda: flow(income, ratio)`) also picks up a
        re-run cell that only redefines `ratio`. A string re-imports the
        module and everything it imports from its directory once a file
        changes. A failed reload is logged as an `Error` and raised; the
        session keeps the pipeline it had. Returns the IPython callback.

        Example::

            s = pipeline.session(df)
            s.watch(lambda: pipeline)
            s.watch("credit.pipeline:pipeline")
        """
        from IPython import get_ipython

        get = ModuleWatcher(source).poll if isinstance(source, str) else source

        def hook(*_: Any) -> None:
            try:
                new = get()
                if new is not None:
                    self.reload(new)
            except Exception as e:
                self._emit(Error(f"reload failed: {type(e).__name__}: {e}", None))
                raise

        get_ipython().events.register("post_run_cell", hook)
        return hook

    def _edit(self, action: str, path: str, new: Step | None) -> Checkpoint | None:
        from decider.steps.base import as_step

        before = self.executable
        root = swap(as_step(before.step), path, new)

        def target(pos: Callable) -> int | None:
            if action == "replace":
                return pos("before", path)
            old_order = list(_order(before.plan.root)[0])
            later = old_order[old_order.index(("after", path)) + 1:]
            return next(i for i in (pos(*e) for e in later) if i is not None)

        return self._apply(root, resolve(root), target, [Edited(action, path)], repr(path))

    def _apply(self, root: Step, plan: Plan, where: Callable, edits: list, label: str,
               new_keys: list | None = None) -> Checkpoint | None:
        from decider.engine import step_map
        from decider.engine.run.engine import Executable

        before, old = self.executable, self.state
        _check_producers(plan, before.plan, label)
        exe = Executable(plan, before.runner, before.lazy, root)
        state, params = exe.prepare(self.frame, self.params)
        # A copy, so a failure below leaves the running one as it was.
        runner = copy.copy(self._runner)
        iterator = runner.iterate(plan, state, params)
        order, spans = _order(plan.root)
        units = {c.id: u for u in getattr(runner, "units", {}).values() if len(u.calls) > 1 for c in u.calls}
        by_path = {c.node.origin.path: c for c in plan.calls}

        def pos(when: str, at: str) -> int | None:
            # A fused kernel is one checkpoint pair: its first call's before, its last call's after.
            unit = units.get(by_path[at].id) if at in by_path else None
            if unit is not None:
                at = unit.calls[0 if when == "before" else -1].node.origin.path
            return order.get((when, at))

        target = where(pos)
        if target is None or target == len(order) - 1:
            # Nothing after it: every node is as it was, so the run ends where it did.
            target = len(order)
        stop = None
        if self.current is not None or self.finished:
            here = None if self.current is None else pos(self.current.when, self.current.origin.path)
            stop = target if here is None else min(target, here)

        skip: dict[Resolved, list[Version]] = {}
        done: list[str] = []
        if stop is not None:
            _skips(plan.root, stop, spans, units, skip, done)
        _carry_over(old, before.plan, state, done)
        cp = None
        if stop is not None:
            runner.skip = skip
            try:
                cp = next((c for c in iterator if pos(c.when, c.origin.path) >= stop), None)
            finally:
                runner.skip = {}

        self.executable, self._runner, self.state, self._params, self._iterator = exe, runner, state, params, iterator
        self._logged = (0, 0, 0)
        self._visits.clear()
        self._index()
        self._keys = new_keys or keys(plan, step_map(root))
        for e in edits:
            self._emit(e)
        if stop is None:
            return None
        if cp is None:
            # The replay reached the end (the edit was at the end, or in an arm no row takes).
            self._next()
            return None
        self.current, self.finished = cp, False
        self._emit(Paused(cp.origin, cp.when, "edit", self._kernels.get(cp.origin.path, ())))
        return cp


def swap(root: Step, path: str, new: Step | None) -> Step:
    """A copy of `root` with the step at `path` replaced by `new` (renamed to fit), or removed if `new` is `None`.

    Steps are frozen, so every parent along `path` is rebuilt; everything
    else is the same object, so its IR comes from the cache.

    Example::

        swap(pipeline, "term/cap_by_income", step(cap_v2))
    """
    if root.name == path:
        if new is None:
            raise ValueError(f"can't delete {path!r}: it is the whole pipeline")
        return new if new.name == path else new.named(path)
    out = _swap(root, root.name or "", path, new)
    if out is root:
        raise KeyError(f"no step at {path!r}")
    return out


def _swap(step: Step, here: str, path: str, new: Step | None) -> Step:
    from decider.steps.base import Step

    changes = {}
    for field in _MEMBERS:
        value = getattr(step, field, None)
        members = value if isinstance(value, tuple) else (value,)
        if not members or not all(isinstance(m, Step) for m in members):
            continue
        out = []
        for m in members:
            at = here if m.name is None else f"{here}/{m.name}" if here else m.name
            if m.name is not None and at == path:
                if new is not None:
                    out.append(new if new.name == m.name else new.named(m.name))
            else:
                out.append(_swap(m, at, path, new))
        if len(out) == len(members) and all(a is b for a, b in zip(out, members)):
            continue
        if not out and not isinstance(value, tuple):
            raise ValueError(f"can't delete {path!r}: it is the {field} of {here!r}; replace it instead")
        changes[field] = tuple(out) if isinstance(value, tuple) else out[0]
    return step._replace(**changes) if changes else step


def _check_producers(plan: Plan, old: Plan, path: str) -> None:
    # A name nothing produces any more reads as an input column, which would fail (or worse, not) at run time.
    orphans = sorted({i.name for i in plan.inputs} & set(old.chains))
    if orphans:
        name = orphans[0]
        reader = next((c.node.origin.path for c in plan.calls for v in c.reads or ()
                       if v.producer is None and v.name == name), "a later step")
        raise WiringError(f"after editing {path} nothing produces {name!r}, which {reader!r} reads; "
                          "keep a step that writes it, or edit the reader too")


def _children(r: Resolved) -> tuple[Resolved, ...]:
    if isinstance(r, Sequence):
        return r.children
    if isinstance(r, Branch):
        return (r.condition, *r.arms)
    if isinstance(r, Loop):
        return (r.condition, r.body)
    return ()


def _order(root: Resolved) -> tuple[dict[tuple[str, str], int], dict[Resolved, tuple[int, int]]]:
    # Every checkpoint in plan order, as if every arm ran and every loop ran once, and each node's span in it.
    order: dict[tuple[str, str], int] = {}
    spans: dict[Resolved, tuple[int, int]] = {}

    def walk(r: Resolved) -> None:
        path = r.node.origin.path
        begin = order[("before", path)] = len(order)
        for child in _children(r):
            walk(child)
        spans[r] = (begin, len(order))
        order[("after", path)] = len(order)

    walk(root)
    return order, spans


def _skips(r: Resolved, stop: int, spans: dict, units: dict, skip: dict, done: list[str]) -> None:
    # Marks the nodes that end before `stop`: the replay passes over them and keeps their earlier values.
    begin, end = spans[r]
    if begin >= stop:
        return
    if isinstance(r, Call) and r.id in units:
        unit = units[r.id]
        if unit.calls[0] is r and spans[unit.calls[-1]][1] < stop:
            skip[r] = [v for v, _ in unit.writes]
            done.extend(c.node.origin.path for c in unit.calls)
        return
    # A frame step of unknown lineage replaces the frame later nodes see, so it runs again.
    barrier = any(isinstance(n, CallNode) and n.kind == "frame" and n.outputs is None for n in iter_nodes(r.node))
    if end < stop and not barrier:
        skip[r] = _passes(r, units)
        done.append(r.node.origin.path)
        return
    # ponytail: a loop around the edit runs again from its first iteration; pass over iterations once runners can seek.
    if not isinstance(r, Loop):
        for child in _children(r):
            _skips(child, stop, spans, units, skip, done)


def _passes(r: Resolved, units: dict) -> list[Version]:
    # The versions a node leaves in sight of the nodes after it.
    if isinstance(r, Call):
        unit = units.get(r.id)
        if unit is None:
            return list(r.writes)
        return [v for v, _ in unit.writes] if unit.calls[0] is r else []
    if isinstance(r, Sequence):
        return [v for c in r.children for v in _passes(c, units)]
    if isinstance(r, Branch):
        return [m.version for m in r.merges]
    return [c.version for c in r.carries]


def _carry_over(old: Any, plan: Plan, state: Any, done: list[str]) -> None:
    # Versions are numbered per plan, so values move across by (name, producer).
    previous = {(v.name, v.producer): v for v in plan.versions}
    for v in state.plan.versions:
        if v.producer is None or any(d in ("", v.producer) or v.producer.startswith(d + "/") for d in done):
            was = previous.get((v.name, v.producer))
            if was is not None and was.id in old.values:
                state.write(v, old.values[was.id], valid=old.valid.get(was.id))
    n = len(plan.versions)
    for v in sorted({v for chain in old.chains.values() for v in chain if v.id >= n}, key=lambda v: v.id):
        state.record(v.name, v.producer, old.values[v.id], old.valid.get(v.id))
