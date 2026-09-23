"""One entry point for every tree document format, with the format detected from the document."""
from __future__ import annotations

import re
import typing as t

from pydantic import BeforeValidator, Discriminator, Tag, TypeAdapter

from decider.steps.trees.schema.conditions import get_field
from decider.steps.trees.schema.flat import FlatRuleDocument, PrioritizedFlatRuleDocument
from decider.steps.trees.schema.v3 import V3TreeDocument

SUPPORTED = ("v3", "flat_rule", "prioritized_flat_rule")


def detect_format(doc: t.Any) -> str:
    """`"v3"`, `"flat_rule"`, `"prioritized_flat_rule"`, or `"v<n>"` for another tree version.

    Flat rules are recognised by `type`; trees by `formatVersion` (or
    `format_version`), then by a `"v<n>-tree"` type. Dict-shaped `nodes` is
    the original v0 format; anything else is read as v3.
    """
    kind = get_field(doc, "type")
    if kind in ("flat_rule", "prioritized_flat_rule"):
        return kind
    version = get_field(doc, "formatVersion")
    if version is None:
        version = get_field(doc, "format_version")
    if version is None and isinstance(kind, str) and (m := re.fullmatch(r"v(\d+)-tree", kind)):
        version = m.group(1)
    if version is None:
        version = 0 if isinstance(get_field(doc, "nodes"), dict) else 3
    return f"v{version}"


def _supported(doc: t.Any) -> t.Any:
    fmt = detect_format(doc)
    if fmt not in SUPPORTED:
        raise ValueError(
            f"{fmt} tree documents are deprecated and no longer supported; "
            "save the tree again as a v3 tree or as flat rules"
        )
    return doc


TreeDocument = t.Annotated[
    t.Union[
        t.Annotated[V3TreeDocument, Tag("v3")],
        t.Annotated[FlatRuleDocument, Tag("flat_rule")],
        t.Annotated[PrioritizedFlatRuleDocument, Tag("prioritized_flat_rule")],
    ],
    Discriminator(detect_format),
    BeforeValidator(_supported),
]
"""A v3 tree, flat-rule or prioritized flat-rule document; `to_tree()` gives the same `Tree` for each."""

_adapter: TypeAdapter = TypeAdapter(TreeDocument)


def load_document(data: t.Any) -> V3TreeDocument | FlatRuleDocument | PrioritizedFlatRuleDocument:
    """Validate a tree document of any supported format.

    Example: `load_document({"type": "flat_rule", "rule": {"rule": {"type": "leaf"}}, "output": {}}).to_tree()`
    """
    return _adapter.validate_python(data)
