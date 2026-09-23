from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from decider.engine.params import NodeParams, ParamsCache, ParamsError, Status, bundle_class, document_key
from decider.registry.resolve import suggest_names

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
        self.key = document_key(doc) if doc else _EMPTY_KEY
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


def check_namespaces(doc: Mapping[str, Any], nodes: dict[int, NodeParams]) -> None:
    """Raise `ParamsError` for any entry of `doc` that names no step with params, or no shared key.

    Example::

        check_namespaces({"kap": {"cap": 12.0}}, nodes)   # ParamsError: ... Did you mean 'cap'?
    """
    if not isinstance(doc, Mapping):
        raise ParamsError(f"a params document is a mapping of step paths, got {type(doc).__name__}")
    paths = {n.path for n in nodes.values()}
    shared = {d.shared_key for n in nodes.values() for d in n.decls if d.shared_key is not None}
    given = doc.get("shared", {})
    for key in given if isinstance(given, Mapping) else ():
        if key not in shared:
            raise ParamsError(f"params document: no step uses shared param '{key}'.{_hint(key, shared)}")
    _walk({k: v for k, v in doc.items() if k != "shared"}, "", paths)


def _walk(level: Mapping[str, Any], prefix: str, paths: set[str]) -> None:
    for key, sub in level.items():
        path = prefix + key
        if path in paths:
            continue
        if isinstance(sub, Mapping) and any(p.startswith(path + "/") for p in paths):
            _walk(sub, path + "/", paths)
            continue
        siblings = {p[len(prefix):].split("/")[0] for p in paths if p.startswith(prefix)}
        raise ParamsError(
            f"params document: no step with params at '{path}'."
            + _hint(key, siblings, prefix)
        )


def _hint(name: str, candidates, prefix: str = "") -> str:
    near = suggest_names(name, candidates)
    return f" Did you mean '{prefix}{near[0]}'?" if near else ""
