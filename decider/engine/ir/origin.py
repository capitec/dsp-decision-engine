from __future__ import annotations

from dataclasses import dataclass

from decider.exceptions import WiringError


@dataclass(frozen=True, slots=True)
class Origin:
    """Where an IR node came from: plain strings, so events and traces carry it as is.

    Args:
        path: the node's id, unique within one IR: names of named steps from
            the root, joined with `/` (`"term/by_sector/cap_private"`).
        source: import path of the code that produced the node
            (`"credit.rules:cap_by_income"`).
        locator: a position inside a `row` node (`"n17"`), shown as `path#n17`.

    Example::

        Origin("term/cap_by_income", "credit.rules:cap_by_income")
    """

    path: str
    source: str
    locator: str | None = None


def check_name(name: str) -> str:
    """Return `name` if it can be a step name in a path, else raise `ValueError`.

    Example::

        check_name("cap_by_income")  # "cap_by_income"
        check_name("a/b")            # ValueError
    """
    if not isinstance(name, str) or not name or "/" in name or "#" in name:
        raise WiringError(
            f"step name {name!r} must be a non-empty string without '/' or '#': "
            "paths join names with '/' and mark positions inside a node with '#'"
        )
    return name
