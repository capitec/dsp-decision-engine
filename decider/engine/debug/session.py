from __future__ import annotations

import copy
from collections import Counter
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
import polars as pl

from decider.engine.debug.commands import Command
from decider.engine.debug.edit import Edits
from decider.engine.debug.events import (Error, Event, NodeFinished, NodeStarted, NodeVisited, Overridden,
                                         ParamsValidated, Paused, RunFinished, RunStarted, Warning, summarize)
from decider.engine.ir.nodes import SequenceNode, iter_nodes
from decider.engine.ir.origin import Origin
from decider.engine.run.runners.base import Checkpoint
from decider.engine.ir.decls import base_annotation
from decider.engine.run.state import dtype_of
from decider.engine.wiring.plan import Version

if TYPE_CHECKING:
    from decider.engine.run.engine import Executable

Breakpoint = str | Callable[[Checkpoint], bool]


class Session(Edits):
    """Run a pipeline one checkpoint at a time: break, inspect, override, resume.

    Nothing runs until a command moves the session. Each command runs
    synchronously and returns the checkpoint it paused at, or `None` once the
    run is over. Everything that happens is appended to `events`.

    Breakpoints: a path (`"term/cap_by_income"`) or prefix (`"term"`) pauses
    just before the first node at or under it, each time the run enters it; `"risk_tree#n17"` pauses just
    after the row node `risk_tree` if any row reached `n17` inside it (only
    runners that report positions, such as the interpreted one); a predicate
    is called with every checkpoint.

    Every mode supports sessions. Interpreted and stepped pause at every
    node. Fused pauses only at kernel boundaries: a kernel running several
    steps is one checkpoint with its first step's path, and a breakpoint (or
    `rewind`) on any of its steps stops just before the whole kernel, with
    the kernel's steps in `Paused.kernel`. Values a kernel uses only inside
    itself are never stored, so `value` and `set` on one raise a `KeyError`
    that suggests `mode="stepped"`.

    `replace(path, step)` and `delete(path)` edit the pipeline mid-run and
    re-run from the edit, keeping every value upstream of it.

    Example::

        s = pipeline.session(df)
        s.break_at("affordability_ratio")
        s.resume()                                # paused before affordability_ratio
        s.value("disposable_income")              # a polars Series
        s.set("disposable_income", 1200.0)        # recorded as override@affordability_ratio
        s.resume()                                # runs to the end
        s.output()["affordability_ratio"]         # computed from 1200.0
        s.state.versions("disposable_income@*")   # the override is in the trail
    """

    def __init__(self, executable: Executable, frame: pl.DataFrame, params: Mapping[str, Any] | None = None):
        self.executable = executable
        self.frame = frame
        self.params = params
        self.events: list[Event] = []
        self.breakpoints: list[Breakpoint] = []
        self.current: Checkpoint | None = None
        self.finished = False
        self._runner = copy.copy(executable.runner)
        if hasattr(self._runner, "visit"):
            self._runner.visit = self._visit
        self._pause = False
        self._visits: Counter[str] = Counter()
        self._visited: set[str] = set()
        self._previous: Checkpoint | None = None
        self._start()
        self._index()
        self._emit(RunStarted(self.state.n))
        self._log_params()

    def break_at(self, target: Breakpoint) -> None:
        """Add a breakpoint: a path, a prefix, `path#locator`, or a predicate on each `Checkpoint`."""
        self.breakpoints.append(target)

    def clear_break(self, target: Breakpoint) -> None:
        """Remove a breakpoint added with the same target."""
        self.breakpoints.remove(target)

    def step(self) -> Checkpoint | None:
        """Advance one node: from just before a call, branch or loop, run it and stop just after it."""
        at = self.current
        if at is not None and at.when == "before" and at.origin.path not in self._sequences:
            return self._go("step", lambda cp: cp.when == "after" and cp.origin == at.origin)
        return self._go("step", lambda cp: True)

    def step_into(self) -> Checkpoint | None:
        """Advance to the very next checkpoint: into a sequence, the taken arm or the next iteration."""
        return self._go("step", lambda cp: True)

    def resume(self) -> Checkpoint | None:
        """Run to the next breakpoint, or to the end."""
        return self._go("breakpoint", lambda cp: False)

    def pause(self) -> None:
        """Stop at the next checkpoint; safe to call while another thread is inside `resume`."""
        self._pause = True

    def set(self, name: str, value: Any) -> None:
        """Override `name`, an input column or a value produced so far, from here on.

        `value` is one value for every row or a list of one per row (`None` is
        null), cast to the name's declared dtype; a failed cast raises
        `ValueError`. Nodes that run later read the override, and it is
        recorded as a version produced by `override@<current path>`.

        Example::

            session.set("disposable_income", 1200.0)
            session.state.versions("disposable_income@*")[-1].producer   # "override@affordability_ratio"
        """
        targets = self._targets(name)
        dtype = dtype_of(base_annotation(targets[-1].annotation))
        values, valid = _cast(name, value, dtype, self.state.n)
        previous = summarize(self.value(name))
        # Every version written so far, not only the latest: a loop or branch
        # may still read an earlier one (a carry, a branch's prior value).
        # ponytail: earlier producers' values are overwritten, not kept; keep copies if the audit needs them.
        # ponytail: frame steps read input columns from the frame, so they miss an input override; overlay inputs too if needed.
        for v in targets:
            self.state.write(v, values.copy(), valid=None if valid is None else valid.copy())
        path = "" if self.current is None else self.current.origin.path
        record = self.state.record(name, f"override@{path}", values, valid)
        self._emit(Overridden(name, record.producer, summarize(self.value(name)), previous))

    def rewind(self, path: str) -> Checkpoint:
        """Re-run from just before the node at (or first under) `path`, keeping every value computed upstream of it.

        Values upstream of `path`, overrides included, are kept as they are
        now; everything from `path` on runs again when the session moves.

        Example::

            session.set("disposable_income", 1200.0)
            session.rewind("affordability_ratio")
            session.resume()
        """
        plan = self.executable.plan
        old = self.state
        saved = self.state, self._params, self._iterator, self._logged
        self._start()
        for cp in self._iterator:
            if cp.when == "before" and self._covers(cp, path):
                break
        else:
            self.state, self._params, self._iterator, self._logged = saved
            raise ValueError(f"no node at or under {path!r} runs")
        # Upstream nodes ran again on the original inputs; the current values win.
        # ponytail: branch routing in the replay comes from the original values; re-route once runners can seek.
        self.state.restore(old, set(self.state.values) | {v.id for v in plan.versions if v.producer is None})
        self._visits.clear()
        self.current, self.finished = cp, False
        self._emit(Paused(cp.origin, cp.when, "rewind", self._kernels.get(cp.origin.path, ())))
        return cp

    def apply(self, command: Command) -> Any:
        """Run a command object (e.g. one decoded from JSON); returns what the matching method returns.

        Example::

            session.apply(SetValue("disposable_income", 1200.0))
        """
        return getattr(self, command.kind)(*(getattr(command, f.name) for f in fields(command) if f.name != "kind"))

    def value(self, spec: str) -> pl.Series:
        """The full current value of `name`, or of `name@path` (the version that node produced).

        Example::

            session.value("term_cap")
            session.value("term_cap@term/cap_by_income")
        """
        if "@" in spec:
            for v in self.state.versions(spec):
                self._check_stored(spec, v)
            return self.state.column(spec)
        return self.state.column(spec, self._targets(spec)[-1])

    def output(self) -> pl.DataFrame:
        """The frame `Executable.run` returns, overrides applied; only once the run has finished."""
        if not self.finished:
            raise RuntimeError("the run hasn't finished; resume() it first")
        return self.executable.output(self.state)

    def _start(self) -> None:
        self.state, self._params = self.executable.prepare(self.frame, self.params)
        self._iterator = self._runner.iterate(self.executable.plan, self.state, self._params)
        self._logged = (0, 0, 0)

    def _index(self) -> None:
        # A fused runner compiles on `iterate`, so call this once the run has started.
        plan = self.executable.plan
        self._sequences = {n.origin.path for n in iter_nodes(plan.root.node) if isinstance(n, SequenceNode)}
        self._produces: dict[str, list[Version]] = {}
        for v in plan.versions:
            self._produces.setdefault(v.producer, []).append(v)
        self._kernels: dict[str, tuple[str, ...]] = {}
        self._internal: dict[int, tuple[str, ...]] = {}
        for unit in getattr(self._runner, "units", {}).values():
            if len(unit.calls) > 1:
                kernel = tuple(c.node.origin.path for c in unit.calls)
                kept = [v for v, _ in unit.writes]
                self._kernels[kernel[0]] = kernel
                self._produces[kernel[0]] = kept
                self._internal.update((v.id, kernel) for c in unit.calls for v in c.writes if v not in kept)

    def _go(self, reason: str, until: Callable[[Checkpoint], bool]) -> Checkpoint | None:
        while (cp := self._next()) is not None:
            if self._pause:
                self._pause, why = False, "pause"
            elif any(self._hits(b, cp) for b in self.breakpoints):
                why = "breakpoint"
            elif until(cp):
                why = reason
            else:
                continue
            self._emit(Paused(cp.origin, cp.when, why, self._kernels.get(cp.origin.path, ())))
            return cp
        return None

    def _next(self) -> Checkpoint | None:
        if self._iterator is None:
            raise RuntimeError("the run is over" if self.finished else "the run stopped on an error; rewind() it")
        try:
            cp = next(self._iterator)
        except StopIteration:
            self._iterator, self.current, self.finished = None, None, True
            self._log_params()
            out = self.executable.output(self.state)
            self._emit(RunFinished({c: summarize(out[c]) for c in out.columns}))
            return None
        except Exception as e:
            self._iterator = None
            self._log_params()
            self._emit(Error(f"{type(e).__name__}: {e}", None if self.current is None else self.current.origin.path))
            raise
        self._previous, self.current = self.current, cp
        self._log_params()
        if cp.when == "before":
            self._emit(NodeStarted(cp.origin))
            return cp
        o = cp.origin
        for locator, rows in self._visits.items():
            self._emit(NodeVisited(Origin(o.path, o.source, locator), rows))
        self._visited = set(self._visits)
        self._visits.clear()
        produced = self._produces.get(o.path, ())
        self._emit(NodeFinished(o, {v.name: summarize(self.state.column(v.name, v)) for v in produced}))
        return cp

    def _hits(self, target: Breakpoint, cp: Checkpoint) -> bool:
        if callable(target):
            return bool(target(cp))
        path, _, locator = target.partition("#")
        if locator:
            return cp.when == "after" and cp.origin.path == path and locator in self._visited
        # Only on entering the subtree, not at every node inside it.
        entering = self._previous is None or not _under(self._previous.origin.path, path)
        return cp.when == "before" and self._covers(cp, path) and entering

    def _covers(self, cp: Checkpoint, prefix: str) -> bool:
        return any(_under(p, prefix) for p in self._kernels.get(cp.origin.path, (cp.origin.path,)))

    def _targets(self, name: str) -> list[Version]:
        # Inputs, then what has been written so far in production order; the
        # override's own records are history, not a place later nodes read.
        inputs = [v for v in self.executable.plan.versions if v.producer is None and v.name == name]
        chain = self.state.chains.get(name, ())
        written = [v for v in chain if v.id in self.state.values and not v.producer.startswith("override@")]
        if not inputs + written:
            # ponytail: only when nothing of `name` is stored; a stored earlier version hides a later in-kernel one.
            for v in chain:
                self._check_stored(name, v)
            raise KeyError(f"{name!r} is neither an input column nor a value produced so far")
        return inputs + written

    def _check_stored(self, spec: str, v: Version) -> None:
        kernel = self._internal.get(v.id)
        if kernel is not None:
            raise KeyError(f"{spec!r} is computed inside the fused kernel {list(kernel)} and never stored; "
                           "open the session with mode='stepped' to inspect or set it")

    def _visit(self, locator: str) -> None:
        self._visits[locator] += 1

    def _log_params(self) -> None:
        report = self._params.report
        n_valid, n_invalid, n_warn = self._logged
        if len(report.validated) > n_valid or len(report.invalid) > n_invalid:
            self._emit(ParamsValidated(tuple(report.validated[n_valid:]), tuple(report.invalid[n_invalid:])))
        for message in report.warnings[n_warn:]:
            self._emit(Warning(message))
        self._logged = (len(report.validated), len(report.invalid), len(report.warnings))

    def _emit(self, event: Event) -> None:
        self.events.append(event)


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _cast(name: str, value: Any, dtype: np.dtype, n: int) -> tuple[np.ndarray, np.ndarray | None]:
    raw = np.empty(n, object)
    try:
        raw[:] = value
    except ValueError:
        raise ValueError(f"can't set {name!r}: give one value or {n} values, not {value!r}") from None
    valid = np.array([v is not None for v in raw], bool)
    if dtype != object:
        try:
            # same_kind: 1.5 into an int column or "abc" into a float one is an error, not a silent truncation.
            raw = np.array([v if ok else 0 for v, ok in zip(raw, valid)]).astype(dtype, casting="same_kind")
        except (TypeError, ValueError):
            raise ValueError(f"can't set {name!r} to {value!r}: it doesn't cast to {dtype}") from None
    return raw, None if valid.all() else valid
