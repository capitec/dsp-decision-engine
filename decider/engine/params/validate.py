from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from pydantic import ValidationError

from decider.engine.params.models import NodeParams
from decider.exceptions import ParamsError
from decider.registry.resolve import hint, suggest


class Status(Enum):
    """Validation status of one node for one params document."""

    UNKNOWN = "unknown"
    OK = "ok"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Validation:
    """The result of validating one node's params.

    `bundle` is the namedtuple passed to the node's function, or `None` when
    `INVALID`. `warnings` lists values replaced by their default under
    `on_invalid="warn"`.
    """

    status: Status
    bundle: tuple | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def check(self, rows: int | None = None) -> Validation:
        """Raise `ParamsError` if invalid; otherwise return self.

        Example::

            bundle = validate_node(node, doc).check(rows=len(df)).bundle
        """
        if self.status is Status.INVALID:
            suffix = f" (affects {rows} row{'' if rows == 1 else 's'})" if rows is not None else ""
            raise ParamsError("invalid params" + suffix + ":\n" + "\n".join(self.errors))
        return self


def _local_view(doc: Mapping, path: str) -> Any:
    view: Any = doc
    for part in path.split("/"):
        view = view.get(part, {}) if isinstance(view, Mapping) else view
    return view


def _message(node: NodeParams, by_name: dict, err: dict) -> str:
    decl = by_name[err["loc"][0]]
    where = f"shared param '{decl.shared_key}'" if decl.shared_key else f"param '{decl.name}'"
    if err["type"] == "missing":
        return f"{node.path}: {where} is required but missing"
    return f"{node.path}: {where}: {err['msg']} (got {err['input']!r})"


def validate_node(node: NodeParams, doc: Mapping) -> Validation:
    """Validate one node's view of a params document.

    The view is the node's own entry (`doc["term"]["cap_by_income"]` for node
    `term/cap_by_income`) plus the `doc["shared"]` keys it uses. Anything
    missing falls back to the node's own default. An invalid value is an error,
    or is replaced by its default when its param says `on_invalid="warn"` or
    `"default"`.

    Example::

        result = validate_node(node, {"term": {"cap_by_income": {"cap": 36.0}}})
        result.status, result.bundle.cap  # Status.OK, 36.0
    """
    local = _local_view(doc, node.path)
    if not isinstance(local, Mapping):
        return Validation(Status.INVALID, None, (f"{node.path}: expected a mapping of params, got {local!r}",))
    shared = doc.get("shared", {})
    errors = []
    by_arg = {d.arg: d.name for d in node.decls if d.shared_key is None and d.arg != d.name}
    for key in local:
        if key not in node.local_names:
            close = suggest(key, node.local_names)
            hint = (f"; did you mean '{by_arg[key]}' (it feeds argument '{key}')?" if key in by_arg
                    else f"; did you mean '{close}'?" if close
                    else f"; its params are {sorted(node.local_names)}")
            errors.append(f"{node.path}: unknown param '{key}'{hint}")
    values = {d.name: local[d.name] for d in node.decls if d.shared_key is None and d.name in local}
    values |= {d.name: shared[d.shared_key] for d in node.decls if d.shared_key is not None and d.shared_key in shared}

    by_name = {d.name: d for d in node.decls}
    warnings = []
    try:
        model = node.model.model_validate(values)
    except ValidationError as e:
        model = None
        for err in e.errors():
            decl, msg = by_name[err["loc"][0]], _message(node, by_name, err)
            if decl.required or decl.on_invalid == "error":
                errors.append(msg)
            else:
                values.pop(decl.name, None)
                if decl.on_invalid == "warn":
                    warnings.append(msg + "; using the default")
    if errors:
        return Validation(Status.INVALID, None, tuple(errors), tuple(warnings))
    if model is None:
        model = node.model.model_validate(values)
    return Validation(Status.OK, node.bundle_type(*(getattr(model, d.name) for d in node.decls)), (), tuple(warnings))


def document_key(doc: Mapping) -> str:
    """A content hash identifying a params document; equal documents share a key.

    Example::

        key = document_key({"shared": {"base_rate": 5.0}})
    """
    # ponytail: JSON content only; key non-JSON values (DataFrames) by identity when table params arrive.
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


class ParamsCache:
    """Validation results per (params document, node path); a document seen before is never revalidated.

    Example::

        cache = ParamsCache()
        key = document_key(doc)
        cache.status(key, node.path)            # Status.UNKNOWN
        bundle = cache.validate(key, doc, node).check().bundle
        cache.status(key, node.path)            # Status.OK
    """

    # ponytail: unbounded; add eviction when a long-lived server sees many distinct documents.
    def __init__(self) -> None:
        self._results: dict[tuple[str, str], Validation] = {}
        self._last: tuple[Any, str] = (None, "")

    def key(self, doc: Mapping) -> str:
        """`document_key(doc)`, hashed once while the same document object is passed call after call.

        The document is treated as immutable: one edited in place and passed
        again keeps its old key, so pass an edited copy instead.

        Example::

            cache.key(doc) == document_key(doc)   # True; the second call with `doc` costs nothing
        """
        # One slot holding the document itself, so its id can't be reused by another object while cached.
        # ponytail: one document per cache; a small dict if callers alternate documents on one executable.
        last = self._last
        if last[0] is doc:
            return last[1]
        key = document_key(doc)
        self._last = (doc, key)
        return key

    def status(self, key: str, path: str) -> Status:
        result = self._results.get((key, path))
        return Status.UNKNOWN if result is None else result.status

    def validate(self, key: str, doc: Mapping, node: NodeParams) -> Validation:
        result = self._results.get((key, node.path))
        if result is None:
            result = self._results[(key, node.path)] = validate_node(node, doc)
        return result


def check_namespaces(doc: Mapping[str, Any], nodes: Mapping[Any, NodeParams]) -> None:
    """Raise `ParamsError` for any entry of `doc` that names no step with params, or no shared key.

    The error lists every step with params and its param names. An empty
    entry (`{}`) is accepted anywhere: it sets nothing.

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
            raise ParamsError(f"params document: no step uses shared param '{key}'.{hint(key, shared)}"
                              f" Shared params: {sorted(shared)}.")
    _walk({k: v for k, v in doc.items() if k != "shared"}, "", paths, nodes)


def _walk(level: Mapping[str, Any], prefix: str, paths: set[str], nodes: Mapping[Any, NodeParams]) -> None:
    for key, sub in level.items():
        path = prefix + key
        # An empty entry sets nothing; templates write one for steps without params.
        if path in paths or (isinstance(sub, Mapping) and not sub):
            continue
        if isinstance(sub, Mapping) and any(p.startswith(path + "/") for p in paths):
            _walk(sub, path + "/", paths, nodes)
            continue
        siblings = {p[len(prefix):].split("/")[0] for p in paths if p.startswith(prefix)}
        raise ParamsError(f"params document: no step with params at '{path}'.{hint(key, siblings, prefix)}"
                          + _expected(nodes))


def _expected(nodes: Mapping[Any, NodeParams]) -> str:
    if not nodes:
        return " This pipeline has no params; pass no params document, or {}."
    steps = sorted(f"{n.path} ({', '.join(d.name for d in n.decls)})" for n in nodes.values())
    return (" The document nests params by step path; steps with params: " + "; ".join(steps)
            + ". pipeline.parameters().defaults() builds a complete document.")
