from __future__ import annotations

import inspect
import typing as t
from abc import ABC

from pydantic import BaseModel

from .resolve import suggest_name


def _declared_tag(cls: type[BaseRegistryModule]) -> t.Any:
    """The `Literal[...]` `cls` declares for its `type` field, or `None` if
    `type` isn't a single-value `Literal` at all.
    """
    field = cls.model_fields.get("type")
    if field is None:
        return None
    annotation = field.annotation
    if t.get_origin(annotation) is not t.Literal:
        return None
    args = t.get_args(annotation)
    return args[0] if len(args) == 1 else None


class BaseRegistryModule(BaseModel, ABC):
    """Base for extension points.

    `class BaseModule(RegistryModule, root=True)` opens an independent
    registry. Any non-abstract subclass anywhere below that root — however
    many layers deep — registers itself automatically under its declared
    `type: t.Literal[...]`; there is no decorator to call. Two hierarchies
    (`BaseModule`, `BaseConfigProvider`, ...) never share entries.

    Registration happens in `__pydantic_init_subclass__`, not the plainer
    `__init_subclass__`: at the point `__init_subclass__` fires, pydantic
    hasn't yet finished building the *subclass's own* `model_fields`, so
    `cls.model_fields["type"]` would still resolve to the parent's `str` —
    `__pydantic_init_subclass__` is pydantic's hook for running code once
    the class (fields included) is actually complete.
    """

    type: str

    _is_registry_root: t.ClassVar[bool] = False
    _registry: t.ClassVar[dict[str, type["BaseRegistryModule"]]]

    def __init_subclass__(cls, root: bool = False, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if root:
            cls._is_registry_root = True
            cls._registry = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        if cls.__dict__.get("_is_registry_root") or inspect.isabstract(cls):
            return

        root_cls = cls._root()
        tag = _declared_tag(cls)
        if tag is None:
            raise TypeError(
                f"{cls.__name__} must declare `type: t.Literal[...]` to "
                f"extend {root_cls.__name__}."
            )
        existing = root_cls._registry.get(tag)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"{tag!r} is already registered to {existing.__name__} on "
                f"{root_cls.__name__} — cannot also register {cls.__name__}."
            )
        root_cls._registry[tag] = cls

    @classmethod
    def _root(cls) -> type["BaseRegistryModule"]:
        for klass in cls.__mro__:
            if klass.__dict__.get("_is_registry_root"):
                return klass
        raise TypeError(
            f"{cls.__name__} has no registry root in its MRO — declare one "
            f"with `class Foo(RegistryModule, root=True)`."
        )

    @classmethod
    def registered_extensions(cls) -> dict[str, type["BaseRegistryModule"]]:
        """`id -> class` for every subclass registered under this root (or,
        called on a leaf, under its root)."""
        return dict(cls._root()._registry)

    @classmethod
    def resolve(cls, tag: str) -> type["BaseRegistryModule"]:
        """The registered class for `tag`, or a `LookupError` naming the
        closest registered tag if `tag` looks like a typo of one."""
        root_cls = cls._root()
        found = root_cls._registry.get(tag)
        if found is not None:
            return found
        known = sorted(root_cls._registry)
        message = f"{tag!r} is not registered on {root_cls.__name__} — registered: {known}."
        hint = suggest_name(tag, known)
        if hint:
            message += f" Did you mean {hint!r}?"
        raise LookupError(message)
