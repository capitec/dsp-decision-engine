from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# check_namespaces is re-exported: it lives with the rest of params validation.
from decider.engine.params import NodeParams, ParamsCache, Status, bundle_class, check_namespaces, document_key  # noqa: F401

_NO_PARAMS = bundle_class(())()
_EMPTY_KEY = document_key({})


@dataclass
class RunReport:
    """What a run did with its params document.

    Args:
        validated: paths of the nodes validated during the run (every node
            with params in eager mode, on a document not seen before; only
            the nodes reached in lazy mode).
        invalid: paths of nodes found invalid.
        warnings: values replaced by their default under `on_invalid="warn"`.

    Example::

        exe.run(df, params=doc)
        exe.report.validated   # ["term/cap_by_income", ...]
    """

    validated: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RunParams:
    """One run's view of a params document: each node's bundle, validated on first use and cached.

    `lazy` says the run validates a node only when it runs.

    Example::

        params = RunParams(nodes, doc, ParamsCache())
        params.bundle(call.id, rows=len(df)).cap
    """

    def __init__(self, nodes: dict[int, NodeParams], doc: Mapping[str, Any], cache: ParamsCache, lazy: bool = False):
        self.nodes = nodes
        self.lazy = lazy
        self.doc = doc
        self.key = cache.key(doc) if doc else _EMPTY_KEY
        self.cache = cache
        self.report = RunReport()

    def bundle(self, call_id: int, rows: int) -> tuple:
        """The params bundle of call `call_id`; raises `ParamsError` naming `rows` if its params are invalid."""
        node = self.nodes.get(call_id)
        if node is None:
            return _NO_PARAMS
        fresh = self.cache.status(self.key, node.path) is Status.UNKNOWN
        result = self.cache.validate(self.key, self.doc, node)
        if fresh:
            self.report.validated.append(node.path)
            self.report.warnings += result.warnings
        if result.status is Status.INVALID:
            self.report.invalid.append(node.path)
        return result.check(rows=rows).bundle

