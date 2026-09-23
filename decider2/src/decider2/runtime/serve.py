"""`Pipeline.serve()` — the long-running-service entry point, doc 03 §6 and
doc 08 §4/§4.1/§4.1b. This is what `serving/` (doc 02 §3.6) is built on top
of, and it is deliberately usable with no server present: nothing here
imports `decider2.serving`, so a pipeline embedded in a notebook or a batch
job can hold a `ServeHandle` with zero HTTP machinery anywhere in the
process.

Doc 02 §3.6 rule 3, verbatim: "the params cell and the generation pointer
live in `runtime/`, not in the handler". `ServeHandle` is that cell plus
that pointer.

**Scope, honestly stated.** decider2 today has no document-driven way to
change a pipeline's *structure* (doc 00-BUILD.md Layer 4, `interiors/`, is
"BLOCKED" — nothing like `ruleset()`/`decision_table()` exists yet). So
every `.stage(doc)` this module can ever actually perform is a **values**
change (doc 08 §2's first row): a params document can retune a threshold,
never add or remove a rule. `ChangeClass.INTERIORS_SHAPE`/`SKELETON` and the
`recompiles=True` path are modelled and guarded (`sealed` mode refuses
them, matching doc 08 §4.1's "zero compilations, ever") so the shape is
right when interiors land, but nothing in this codebase can produce them
today — see this agent's report for the full note.

This module intentionally does not import `decider2.graph`: it only calls
methods a `Pipeline` already exposes (`.elements`, `.interface`, `.emits`,
`.flatten_for_runtime()`, `.score()`), the same one-way-dependency
discipline `decider2.runtime.invoke`'s own docstring states ("this module
does not import `decider2.graph` itself").
"""
from __future__ import annotations

import difflib
import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "ChangeClass",
    "StagePlan",
    "ServeHandle",
    "SealedModeError",
    "structure_fingerprint",
]

_MODES = ("sealed", "live")


class ChangeClass(Enum):
    """Doc 08 §2 / §4.1b's three change classes. `resolve()`/`.stage()`
    below only ever produces `VALUES` today — see the module docstring."""

    VALUES = "values"
    INTERIORS_SHAPE = "interiors_shape"
    SKELETON = "skeleton"


@dataclass(frozen=True)
class StagePlan:
    """What `handle.stage(doc)` hands back, doc 08 §4.1b verbatim:

        plan.klass          # ChangeClass.VALUES | INTERIORS_SHAPE | SKELETON
        plan.recompiles     # False | True
        plan.fingerprint    # the new structure fingerprint
        plan.eta            # ~0 s, or a compile estimate

    `doc` is the fully-merged override document this plan would activate —
    private to this module (not part of doc 08's sketch), carried here
    rather than recomputed so `.activate()` cannot desync from what
    `.stage()` actually validated.
    """

    klass: ChangeClass
    recompiles: bool
    fingerprint: str
    eta: float
    doc: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, repr=False)
    shared: Mapping[str, Any] | None = field(default=None, repr=False)


class SealedModeError(RuntimeError):
    """Doc 08 §4.1: `sealed` mode's `--verify` guarantee is "no compilation
    occurs after `.warm()` has run" (revised — see this module's own report
    for why "zero, ever" stopped being literally achievable, and doc 05 §8
    for the updated text). Raised by `.stage()`/`.activate()` for a plan
    that would recompile while a handle is in `sealed` mode. Unreachable
    today (see module docstring) — kept so the guarantee is enforced in
    code the day a doc-driven structure change exists, not left as a
    comment."""


def structure_fingerprint(pipeline: Any) -> str:
    """Doc 08 §4.1b / §8: "a content hash over the structural elements
    only — rule shape, operators, wiring, nesting — with every *value*
    excluded." Mirrors `compile.driver._driver_key`'s own rule that a
    compiled artefact's identity never depends on a params *value* — this
    is the same idea one layer up, over the graph rather than the compiled
    kernel.

    Reads only `Step.name`/`.inputs`/`.params` (names and annotations, never
    a `ParamDecl.default`), `.reads_params`/`.reads_shared`/`.nogil`, and
    each `Module`'s name and `.bound` *keys* (never bound *values*) — so
    retuning a threshold, including one frozen by `.bind()`, never changes
    this hash.
    """
    parts: list[Any] = []
    for m in pipeline.elements:
        step_parts = []
        for s in m.steps:
            inputs = tuple((i.name, str(i.annotation), i.null_policy.value) for i in s.inputs)
            params = tuple((p.name, str(p.annotation)) for p in s.params)
            step_parts.append(
                (s.name, inputs, params, s.reads_params, s.reads_shared, s.nogil)
            )
        parts.append((m.name, tuple(step_parts), tuple(sorted(m.bound))))
    blob = json.dumps(parts, sort_keys=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _unknown_module_error(key: str, by_name: Mapping[str, Any]) -> ValueError:
    tunable = sorted(name for name, m in by_name.items() if m.params_model is not None)
    near = difflib.get_close_matches(key, tunable, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    return ValueError(
        f"params document has no module instance {key!r} ({len(tunable)} "
        f"tunable: {', '.join(tunable) or 'none'}).{hint} Params are "
        "namespaced by module instance name (doc 03 §10)."
    )


class ServeHandle:
    """`handle = pipeline.serve()` (doc 03 §6). Owns:

    - the **params generation pointer** — the currently-active override
      document, read once per `.score()` call;
    - a small history of previous generations, for free `.rollback()`
      (doc 08 §4 property 5);
    - the pipeline's **structure fingerprint** (doc 08 §4.1b), computed
      once at construction since nothing in this codebase can change a
      served pipeline's structure after the fact (see module docstring).

    Not thread-safe by accident: `serving/server.py`'s stdlib fallback runs
    on a `ThreadingHTTPServer`, so every mutation below takes `_lock`.
    """

    def __init__(self, pipeline: Any, *, mode: str = "sealed") -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.pipeline = pipeline
        self.mode = mode
        self._fingerprint = structure_fingerprint(pipeline)
        self._lock = threading.RLock()
        self._current: dict[str, dict[str, Any]] = {}
        self._shared: dict[str, Any] | None = None
        self._history: list[tuple[dict, dict | None]] = []
        self._pending: StagePlan | None = None
        self._warm_report: Any | None = None  # PrecompileReport, once warm() runs

    # --- read-only facts about this handle ----------------------------------

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def pending(self) -> StagePlan | None:
        return self._pending

    @property
    def generations(self) -> int:
        """How many params generations are resident: the active one plus
        whatever `.rollback()` can still reach. Doc 08 §4's "holding three
        generations resident" is about the *compiled artefact* when a
        structure change is involved; every generation here shares the one
        compiled `Driver` (see module docstring), so this counts retained
        params snapshots, not distinct compiled code."""
        return len(self._history) + 1

    @property
    def is_warm(self) -> bool:
        """`True` once `.warm()` has run. `serving/dispatch.py`'s `/ping`
        checks this and answers 503 (not 200) until it is — doc 05 §8's
        "zero compilations" guarantee only holds AFTER warm-up (this
        module's own report on why "zero, ever" stopped being literally
        true); a `/ping` that says 200 before that would let a request
        arrive first and pay the compile itself, which is the exact
        failure this property exists to prevent."""
        return self._warm_report is not None

    def warm(self, *, shared: dict | None = None) -> Any:
        """Force every numba specialisation this handle's `score()`/
        `apply()` calls will need, once, before either is ever asked to
        answer a real request — `Pipeline.precompile()`, called with
        whatever `shared=` this deployment's tables (doc 08 §3.4) need.
        Idempotent: calling it again just re-measures and re-stores the
        report (harmless — every kernel it would compile is, by
        definition, already compiled the second time).
        """
        with self._lock:
            self._shared = dict(shared) if shared else self._shared
            report = self.pipeline.precompile(shared=self._shared)
            self._warm_report = report
            return report

    # --- the parameter-play surface -----------------------------------------

    def params_schema(self) -> dict[str, Any]:
        """Doc 08 §6.2's "JSON Schema — the UI contract", one entry per
        module that has params to tune. `model_json_schema()` is pydantic's
        own — nothing here hand-rolls a schema shape."""
        return {
            m.name: m.params_model.model_json_schema()
            for m in self.pipeline.elements
            if m.params_model is not None
        }

    def _effective(self, module: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
        merged = {**dict(module.bound), **dict(raw)}
        return module.params_model(**merged).model_dump()

    def resolved_params(self) -> dict[str, Any]:
        """The fully-resolved current document: every tunable module, every
        field, defaults filled in — "the current resolved params document"
        the task asks `GET /params` to answer. Posting this exact document
        straight back to `.stage()` is a no-op by construction, which is
        what makes a GET-then-POST round trip idempotent."""
        out: dict[str, Any] = {}
        for m in self.pipeline.elements:
            if m.params_model is None:
                continue
            out[m.name] = self._effective(m, self._current.get(m.name, {}))
        if self._shared:
            out["shared"] = dict(self._shared)
        return out

    def _validate(
        self, doc: Mapping[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
        """Validate `doc` module by module. Raises `pydantic.ValidationError`
        (unknown field, out-of-bounds value — pydantic's own message) or
        `ValueError` (unknown module namespace, a `.bind()`-frozen field) —
        never anything else, so a caller one layer up can treat both as
        "this is a 400" without inspecting the exception further. Mirrors
        `runtime.invoke.resolve_params`'s own checks (doc 03 §10), because a
        document this accepts must behave identically once it actually
        reaches that function at score time.
        """
        by_name = {m.name: m for m in self.pipeline.elements}
        validated: dict[str, dict[str, Any]] = {}
        shared: dict[str, Any] | None = None
        for key, value in doc.items():
            if key == "shared":
                if not isinstance(value, Mapping):
                    raise ValueError("params['shared'] must be an object")
                shared = dict(value)
                continue
            if key not in by_name:
                raise _unknown_module_error(key, by_name)
            module = by_name[key]
            if module.params_model is None:
                raise ValueError(
                    f"module '{key}' declares no params to override (doc 03 §4)."
                )
            if not isinstance(value, Mapping):
                raise ValueError(f"params[{key!r}] must be an object")
            frozen = sorted(set(value) & set(module.bound))
            if frozen:
                raise ValueError(
                    f"params[{key!r}] sets {frozen}, which "
                    f"{'is' if len(frozen) == 1 else 'are'} frozen by "
                    f"{key}.bind(). A bound value left the caller-facing "
                    "params interface (doc 03 §4.3) — unbind it at "
                    "composition, or drop it from the request."
                )
            # Raises pydantic.ValidationError on an unknown field or an
            # out-of-bounds value — that IS the check; the resolved value
            # itself is only needed by `resolved_params()`.
            self._effective(module, value)
            validated[key] = dict(value)
        return validated, shared

    def stage(self, doc: Mapping[str, Any]) -> StagePlan:
        """Doc 08 §4.1b / doc 03 §6. Validates `doc`, merges it into the
        currently-active document at module-key granularity (a module
        named in `doc` has its whole override replaced — unmentioned
        modules are untouched), and returns the plan without activating
        anything. Doc 08's "the framework never polls" holds here too:
        this only runs when a caller calls it.
        """
        validated, shared = self._validate(doc)
        with self._lock:
            new_current = {**self._current, **validated}
            new_shared = self._shared if shared is None else shared
            # Doc 08 §4.2: every field this codebase can validate through a
            # pydantic model is a VALUE of an already-compiled type — see
            # the module docstring for why INTERIORS_SHAPE/SKELETON can
            # never actually be produced here yet.
            klass = ChangeClass.VALUES
            recompiles = False
            plan = StagePlan(
                klass=klass, recompiles=recompiles, fingerprint=self._fingerprint,
                eta=0.0, doc=new_current, shared=new_shared,
            )
            self._enforce_sealed(plan)
            self._pending = plan
        return plan

    def _enforce_sealed(self, plan: StagePlan) -> None:
        if plan.recompiles and self.mode == "sealed":
            raise SealedModeError(
                "this handle is serving in 'sealed' mode (doc 08 §4.1: "
                "'no compilation after warm-up'), and this change recompiles "
                f"({plan.klass.value}). Serve in mode='live' to allow it."
            )

    def activate(self) -> StagePlan:
        """Doc 03 §6 / doc 08 §4 property 4: "activation is explicit".
        3.36 µs in doc 08's own measurement, because this is nothing more
        than swapping which dict `.score()` reads — no compile is ever
        involved for a document this codebase can produce (module
        docstring)."""
        with self._lock:
            if self._pending is None:
                raise RuntimeError(
                    "nothing staged — call .stage(doc) before .activate()."
                )
            plan = self._pending
            self._history.append((self._current, self._shared))
            self._current = dict(plan.doc)
            self._shared = dict(plan.shared) if plan.shared else None
            self._pending = None
        return plan

    def rollback(self) -> dict[str, Any]:
        """Doc 08 §4 property 5: "rollback is free" — the previous
        generation never stopped being valid, so this is just popping the
        history stack. Returns the restored *raw* override document (the
        same shape `.stage()` takes)."""
        with self._lock:
            if not self._history:
                raise RuntimeError("no previous generation to roll back to.")
            self._current, self._shared = self._history.pop()
            self._pending = None
            return dict(self._current)

    # --- serving the actual decision ----------------------------------------

    def score(
        self, record: Mapping[str, Any], *, params: Mapping[str, Any] | None = None,
        shared: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Doc 08 §4 property 1: read the generation pointer exactly once.
        `params=`/`shared=` let a caller (namely `.preview()`) score against
        something other than the active generation without ever touching
        it."""
        with self._lock:
            doc = dict(self._current) if params is None else dict(params)
            sh = self._shared if shared is None else shared
        return self.pipeline.score(record, params=doc, shared=sh)

    def preview(
        self, record: Mapping[str, Any], proposed: Mapping[str, Any], *,
        shared: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """`POST /params/preview`'s engine: validate `proposed` exactly as
        `.stage()` would, then score the SAME record against the active
        generation and the proposed one, side by side — without staging or
        activating either. Doc 02 §3.6: this is the "play with it"
        affordance the whole task is organised around.
        """
        validated, proposed_shared = self._validate(proposed)
        with self._lock:
            current_doc = dict(self._current)
            proposed_doc = {**self._current, **validated}
            base_shared = self._shared if shared is None else shared
            merged_shared = base_shared if proposed_shared is None else proposed_shared
        current = self.pipeline.score(record, params=current_doc, shared=base_shared)
        proposed_out = self.pipeline.score(record, params=proposed_doc, shared=merged_shared)
        return {"current": current, "proposed": proposed_out}

    # --- doc 00 §2c: report which kernels hold the GIL, never decide -------

    def gil_report(self) -> list[dict[str, Any]]:
        """One entry per compiled kernel (fuse()-group) or fallback
        segment — each segment builds its own entry
        (`Segment.gil_report_entry`, `decider2.compile.driver`) rather than
        this method branching on what kind it is. `holds_gil=True` means at
        least one step in that kernel did not declare `@step(nogil=True)`
        — doc 00 §2c: "a group releases the GIL only when every step in it
        asked to". A fallback segment runs in plain Python and always
        holds the GIL.

        Builds/loads the driver with the exact same key
        (`steps`/`group_ids`/`owners`/`build_dir`/`terminal_names`) that
        `Pipeline.score()`/`.apply()` build internally, so this reports on
        the actual compiled artefact serving requests — not a shadow
        compile of its own — via `compile.driver.build_driver`'s existing
        cache.
        """
        from decider2.compile.driver import build_driver
        from decider2.runtime.invoke import DEFAULT_BUILD_DIR

        steps, group_ids, owners, _ = self.pipeline.flatten_for_runtime()
        emit_names = frozenset(e.name for e in getattr(self.pipeline, "emits", ()))
        terminal_names = frozenset(self.pipeline.interface.terminals) | emit_names
        driver = build_driver(
            list(steps), list(group_ids), owners=list(owners),
            build_dir=DEFAULT_BUILD_DIR, terminal_names=terminal_names,
        )
        return [seg.gil_report_entry() for seg in driver.segments]

    def health(self) -> dict[str, Any]:
        """`GET /health`'s body (doc 00 §2c + doc 08 §4.1b)."""
        with self._lock:
            generations = self.generations
            warm = self.is_warm
        return {
            "status": "ok",
            "mode": self.mode,
            "fingerprint": self._fingerprint,
            "generations": generations,
            "kernels": self.gil_report(),
            "warm": warm,
        }
