"""A `RequestHandler` that also hands back the typed record it scored, and can score with other params."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from decider.serving import RequestHandler


class _Tap:
    """Wraps an executable to remember the last record it scored."""

    def __init__(self, inner, owner):
        self._inner, self._owner = inner, owner

    def score(self, record, params):
        self._owner.typed = record
        return self._inner.score(record, params)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class Replayer(RequestHandler):
    """Scores raw JSON requests exactly as `decider serve` does, against one staged config version.

    `params` replaces the version's params document, for what-if runs.

        replayer.score({"decision_id": "a", ...})   # the response as a dict
    """

    params: dict | None = None
    typed: dict | None = field(default=None, init=False, repr=False)

    def module_fn(self):
        live = super().module_fn()
        params = live.params if self.params is None else self.params
        return live._replace(params=params, executable=_Tap(live.executable, self))

    def score(self, request: dict) -> dict:
        body = json.dumps(request).encode()
        return json.loads(asyncio.run(self.process_fn(body, "application/json", "application/json")).content)

    def untapped(self):
        return RequestHandler.module_fn(self)
