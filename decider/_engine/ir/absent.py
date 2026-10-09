"""A `NotRequired` key that a frame column holds as null, which a single record would not have."""
from __future__ import annotations

from typing import Any, get_args, get_origin

import typing_extensions as te

from decider._engine.ir.decls import base_annotation, nullable
from decider._engine.types import annotation_cache


@annotation_cache
def has_optional_key(annotation: Any) -> bool:
    """Whether `annotation` is, or holds, a TypedDict with a `NotRequired` key."""
    if te.is_typeddict(annotation):
        return bool(annotation.__optional_keys__) or any(has_optional_key(h) for h in te.get_type_hints(annotation).values())
    return any(has_optional_key(a) for a in get_args(annotation))


def without_absent(value: Any, annotation: Any) -> Any:
    """`value` with the null of every `NotRequired` key (not declared `| None`) removed, at any depth.

    Polars gives every struct the union of the keys its rows had, so a key one item left out reads back as null.
    """
    a = base_annotation(annotation)
    if isinstance(value, list) and get_origin(a) is list:
        return [without_absent(v, get_args(a)[0]) for v in value]
    if isinstance(value, dict) and te.is_typeddict(a):
        hints = te.get_type_hints(a)
        return {k: without_absent(v, hints[k]) if k in hints else v for k, v in value.items()
                if not (v is None and k in a.__optional_keys__ and not nullable(hints[k]))}
    return value
