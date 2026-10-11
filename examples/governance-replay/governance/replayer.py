"""A `RequestHandler` that also hands back the typed record it scored, and can score with other params."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from decikit.serving import RequestHandler


class _Tap:
    """Wraps an executable to remember the last record it scored."""

    def __init__(self, inner, owner):
        self._inner, self._owner = inner, owner

    def run(self, data, params):
        if isinstance(data, dict):
            self._owner.typed = data
        return self._inner.run(data, params)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class Replayer(RequestHandler):
    """Scores raw JSON requests exactly as `decikit serve` does, against one staged config version.

    `params` replaces the version's params document, for what-if runs. `run` gives the typed result;
    `served` gives it as the client would receive it, which is what evidence records and replay compares.

        replayer.served({"decision_id": "a", ...})   # the response as a JSON-safe dict
    """

    params: dict | None = None
    typed: dict | None = field(default=None, init=False, repr=False)

    def module_fn(self):
        live = super().module_fn()
        params = live.params if self.params is None else self.params
        return live._replace(params=params, executable=_Tap(live.executable, self))

    def served(self, request: dict) -> dict:
        return json.loads(self.output_fn(self.run(request), "application/json").content)

    def untapped(self):
        return RequestHandler.module_fn(self)
