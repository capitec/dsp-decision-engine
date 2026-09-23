"""Framework-agnostic request handling — doc 02 §3.6's seam, made concrete.

Every route a serving backend needs is one small `Dispatcher` method that
takes and returns plain dicts and raises `BadRequest` for a client error.
Wiring a *new* transport (Flask, FastAPI, aiohttp, a Lambda handler, decider
1's own sanic backend) means writing the thin adapter in `app.py` — never
touching this file, and never touching `pipeline`/`handle` themselves. This
is what doc 02 §3.6 means by "someone should be able to write their own
serving layer pretty easily": the seam is here, not in any one HTTP
framework's routing decorators.

`Dispatcher` owns no state of its own — the params generation pointer and
the structure fingerprint live on the `ServeHandle` (doc 02 §3.6 rule 3),
so restarting a transport (or running two side by side, e.g. a debug port
and a public one) never risks two copies of that state disagreeing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from decider2.runtime.serve import SealedModeError, ServeHandle

__all__ = ["Dispatcher", "BadRequest", "ROUTES"]


class BadRequest(Exception):
    """A client error — every serving backend in this package turns this
    into a 400, never a 500 (the task's own requirement: "validation
    errors come back as 400 with pydantic's own message, not a 500")."""

    def __init__(self, message: str, *, errors: list | None = None):
        super().__init__(message)
        self.errors = errors


def _as_bad_request(exc: Exception) -> BadRequest:
    if isinstance(exc, ValidationError):
        return BadRequest(str(exc), errors=exc.errors())
    return BadRequest(str(exc))


@dataclass
class Dispatcher:
    """Wraps one `ServeHandle`. Every method returns `(status_code, body)`;
    every method raises `BadRequest` (never lets a `ValidationError`/
    `ValueError` escape as-is) for anything a caller got wrong."""

    handle: ServeHandle

    # --- SageMaker convention (doc 02 §3.6) ---------------------------------

    def ping(self, _body: Any = None) -> tuple[int, dict]:
        """200 iff the model is loaded AND warm — `handle.is_warm`
        (`ServeHandle.warm()`, doc 05 §8's revised guarantee). Reaching
        this code used to be treated as proof enough (the pipeline is
        compiled/compilable, the handle exists) — it is not: numba compiles
        a kernel lazily, on its own first real call, unless something
        already forced it. `serving/app.py`'s `app()` calls `handle.warm()`
        before ever returning an app a server can bind and accept
        connections on, so in the ordinary path this is already `True` by
        the time anything can reach `/ping` at all; this check exists for a
        caller that built the app with `warm=False` or is racing warm-up
        some other way — 503, not 200, until it finishes, so a request
        never lands on the first-real-call compile `precompile()` exists to
        avoid."""
        if not self.handle.is_warm:
            return 503, {"status": "warming up"}
        return 200, {"status": "ok"}

    def invocations(self, record: dict) -> tuple[int, dict]:
        """One record in, one decision out. `record` must be a JSON object
        — `pipeline.score(dict)`, never kwargs (doc 03 §6, EXPERIMENTS.md
        §N2)."""
        if not isinstance(record, dict):
            raise BadRequest("/invocations expects a JSON object (one record)")
        try:
            decision = self.handle.score(record)
        except (ValidationError, ValueError) as exc:
            raise _as_bad_request(exc) from exc
        return 200, decision

    # --- the parameter-play surface (doc 03 §6, doc 08 §4) ------------------

    def get_params(self, _body: Any = None) -> tuple[int, dict]:
        return 200, self.handle.resolved_params()

    def get_params_schema(self, _body: Any = None) -> tuple[int, dict]:
        return 200, self.handle.params_schema()

    def post_params(self, doc: dict) -> tuple[int, dict]:
        """Stage AND activate in one call — the task's own spelling of this
        endpoint. The two-step primitive (`handle.stage()`/`.activate()`,
        doc 03 §6) is still there for a caller that wants to inspect the
        plan before committing; this route is the one-shot convenience over
        it."""
        if not isinstance(doc, dict):
            raise BadRequest("POST /params expects a JSON object (a params document)")
        try:
            plan = self.handle.stage(doc)
            self.handle.activate()
        except (ValidationError, ValueError, SealedModeError) as exc:
            raise _as_bad_request(exc) from exc
        return 200, {
            "change_class": plan.klass.value,
            "recompiles": plan.recompiles,
            "fingerprint": plan.fingerprint,
            "eta": plan.eta,
            "params": self.handle.resolved_params(),
        }

    def preview(self, payload: dict) -> tuple[int, dict]:
        """`{"record": {...}, "params": {...}, "shared": {...}?}` in;
        `{"current": {...}, "proposed": {...}}` out — doc's own "play with
        it" affordance: see what a change would do before committing."""
        if not isinstance(payload, dict):
            raise BadRequest("POST /params/preview expects a JSON object")
        record = payload.get("record")
        if not isinstance(record, dict):
            raise BadRequest("POST /params/preview needs a 'record' object to score")
        proposed = payload.get("params", {})
        if not isinstance(proposed, dict):
            raise BadRequest("POST /params/preview's 'params' must be an object")
        shared = payload.get("shared")
        try:
            result = self.handle.preview(record, proposed, shared=shared)
        except (ValidationError, ValueError) as exc:
            raise _as_bad_request(exc) from exc
        return 200, result

    def rollback(self, _body: Any = None) -> tuple[int, dict]:
        try:
            self.handle.rollback()
        except RuntimeError as exc:
            raise BadRequest(str(exc)) from exc
        return 200, {"params": self.handle.resolved_params()}

    # --- doc 00 §2c ----------------------------------------------------------

    def health(self, _body: Any = None) -> tuple[int, dict]:
        return 200, self.handle.health()


# (method, path) -> Dispatcher method name. `app.py`'s two backends both
# route through this one table, so adding an endpoint means adding one line
# here plus one `Dispatcher` method — never touching either transport.
ROUTES: dict[tuple[str, str], Callable[[Dispatcher, Any], tuple[int, dict]]] = {
    ("GET", "/ping"): Dispatcher.ping,
    ("POST", "/invocations"): Dispatcher.invocations,
    ("GET", "/params"): Dispatcher.get_params,
    ("GET", "/params/schema"): Dispatcher.get_params_schema,
    ("POST", "/params"): Dispatcher.post_params,
    ("POST", "/params/preview"): Dispatcher.preview,
    ("POST", "/rollback"): Dispatcher.rollback,
    ("GET", "/health"): Dispatcher.health,
}
