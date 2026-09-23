from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from pydantic import ValidationError

from decider.engine.params.models import NodeParams
from decider.exceptions import ParamsError
from decider.registry.resolve import suggest


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

    def status(self, key: str, path: str) -> Status:
        result = self._results.get((key, path))
        return Status.UNKNOWN if result is None else result.status

    def validate(self, key: str, doc: Mapping, node: NodeParams) -> Validation:
        result = self._results.get((key, node.path))
        if result is None:
            result = self._results[(key, node.path)] = validate_node(node, doc)
        return result
