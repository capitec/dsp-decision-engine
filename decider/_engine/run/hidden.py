from __future__ import annotations

import warnings

from decider._engine.compile import hidden_state
from decider._engine.wiring import Plan
from decider.exceptions import EngineError, HiddenStateWarning

_ADVICE = (" decider.lab.diff cannot compare an object, so a release check cannot tell \"unchanged\" from \"the limit "
           "doubled\", and numba cannot compile the step. Put the data in a config document or a "
           "ConfigurableStep/CustomFunction field and call a plain function, or use param()/param_table(). "
           "See `decider guide`, \"Shipping a step type in a package\".")


def warn_hidden_state(plan: Plan, strict: bool = False) -> None:
    """Warn once per hidden object each scalar or row step of `plan` reads; with `strict`, raise instead."""
    for call in plan.calls:
        if call.node.kind == "frame":
            continue
        for name, _, type_name, site in hidden_state(call.node.fn):
            message = f"{call.node.origin.path} reads {type_name} `{name}` ({site}).{_ADVICE}"
            if strict:
                raise EngineError(message)
            warnings.warn(message, HiddenStateWarning, stacklevel=3)
