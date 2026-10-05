from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Source:
    """A step function's parsed source: the tree, its first line, and each argument's declared annotation."""

    tree: ast.AST
    start: int
    file: str | None
    inputs: dict[str, Any]

    def line(self, node: ast.AST) -> int:
        return self.start + node.lineno - 1


def parse(fn: Any, inputs: dict[str, Any]) -> Source | None:
    try:
        text, start = inspect.getsourcelines(fn)
        file = inspect.getsourcefile(fn)
    except (OSError, TypeError):
        return None
    # getsourcelines keeps a nested function's indentation, which ast.parse rejects.
    return Source(ast.parse(textwrap.dedent("".join(text))), start, file, inputs)
