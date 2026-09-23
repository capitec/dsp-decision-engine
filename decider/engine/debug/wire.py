from __future__ import annotations

from typing import Annotated

from pydantic import Field, TypeAdapter

from decider.engine.debug.commands import Command
from decider.engine.debug.events import Event

EVENT = TypeAdapter(Annotated[Event, Field(discriminator="kind")])
"""JSON for one event.

Example::

    EVENT.validate_json(EVENT.dump_json(event)) == event
"""

EVENT_LOG = TypeAdapter(list[Annotated[Event, Field(discriminator="kind")]])
"""JSON for a list of events, such as `session.events`.

Example::

    EVENT_LOG.validate_json(EVENT_LOG.dump_json(session.events)) == session.events
"""

COMMAND = TypeAdapter(Annotated[Command, Field(discriminator="kind")])
"""JSON for one command.

Example::

    session.apply(COMMAND.validate_json('{"kind": "set", "name": "x", "value": 1.0}'))
"""
