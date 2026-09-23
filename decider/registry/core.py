from __future__ import annotations

import importlib
import inspect
import typing as t
from abc import ABC

from pydantic import BaseModel, SerializeAsAny, WrapValidator, model_validator

from decider.exceptions import RegistryError

from .resolve import hint

R = t.TypeVar("R", bound="BaseRegistryModule")


def import_path(cls: type) -> str:
    """`"<module>:<QualName>"`, the tag every registered class answers to."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _alias(cls: type[BaseRegistryModule]) -> str | None:
    annotation = cls.model_fields["type"].annotation
    if t.get_origin(annotation) is not t.Literal:
        return None
    args = t.get_args(annotation)
    if len(args) != 1 or not isinstance(args[0], str):
        raise TypeError(f"{import_path(cls)}: `type` must be a single-string `Literal`.")
    return args[0]


class BaseRegistryModule(BaseModel, ABC):
    """Base for pydantic classes that config addresses by a `type` tag.

    `class Root(BaseRegistryModule, root=True)` opens an independent registry.
    Every non-abstract subclass below it registers itself on definition under
    its import path `"<module>:<QualName>"`, and under an alias too if it
    declares `type: Literal["alias"]`. Either spelling loads it; dumps write
    the alias when there is one, else the import path.

    Example:
        class Step(BaseRegistryModule, root=True):
            name: str

        class Tree(Step):
            type: t.Literal["tree"] = "tree"

        Step.resolve("tree") is Step.resolve(import_path(Tree)) is Tree
        Tree(name="t").model_dump()  # {"type": "tree", "name": "t"}
    """

    type: str

    _is_registry_root: t.ClassVar[bool] = False
    _registry: t.ClassVar[dict[str, type[BaseRegistryModule]]]
    _tag: t.ClassVar[str | None] = None

    def __init_subclass__(cls, root: bool = False, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if root:
            cls._is_registry_root = True
            cls._registry = {}

    # Fields are only final here, not in `__init_subclass__`.
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if cls.__dict__.get("_is_registry_root") or inspect.isabstract(cls):
            return
        path = import_path(cls)
        alias = _alias(cls)
        registry = cls._root()._registry
        if alias is not None:
            existing = registry.get(alias)
            if existing is not None and import_path(existing) != path:
                raise ValueError(
                    f"type alias {alias!r} is claimed by both {import_path(existing)} and {path}; "
                    "give one its own `type` (a new `Literal` alias, or `str` for the import path)."
                )
            registry[alias] = cls
        # Same import path means the class was redefined (a notebook reload): replace it.
        registry[path] = cls
        cls._tag = alias or path

    @model_validator(mode="before")
    @classmethod
    def _normalise_type(cls, data: t.Any) -> t.Any:
        if cls._tag is None or not isinstance(data, dict):
            return data
        tag = data.get("type", cls._tag)
        if tag not in (cls._tag, import_path(cls)):
            raise ValueError(f"type {tag!r} does not match {cls._tag!r}.")
        return {**data, "type": cls._tag}

    @classmethod
    def _root(cls) -> type[BaseRegistryModule]:
        for klass in cls.__mro__:
            if klass.__dict__.get("_is_registry_root"):
                return klass
        raise TypeError(
            f"{cls.__name__} has no registry root; declare one with "
            "`class Root(BaseRegistryModule, root=True)`."
        )

    @classmethod
    def resolve(cls: type[R], tag: str) -> type[R]:
        """The registered subclass of `cls` for an alias or import path.

        An import path whose module isn't loaded yet is imported first. Only a
        class that registered under this root is ever returned, so a config
        can't name arbitrary code (`"os:system"` is rejected).

        Raises:
            RegistryError: (a `LookupError`) unknown tag, with a did-you-mean when one is close.
        """
        registry = cls._root()._registry
        if tag not in registry and ":" in tag:
            try:
                importlib.import_module(tag.partition(":")[0])
            except ImportError:
                pass
        found = registry.get(tag)
        if found is not None and issubclass(found, cls):
            return found
        known = sorted(k for k, c in registry.items() if issubclass(c, cls))
        raise RegistryError(f"{tag!r} is not a registered {cls.__name__} type.{hint(tag, known)}")

    @classmethod
    def json_schema(cls) -> dict[str, dict[str, t.Any]]:
        """`{tag: JSON schema}` for every registered subclass of `cls`, for UI forms."""
        classes = {c._tag: c for c in cls._root()._registry.values() if issubclass(c, cls)}
        schemas = {}
        for tag, klass in sorted(classes.items()):
            schema = klass.model_json_schema()
            schema["properties"]["type"] = {"const": tag, "default": tag, "type": "string"}
            schemas[tag] = schema
        return schemas


def ref(root: type[R]) -> t.Any:
    """A field type holding any registered subclass of `root`, chosen by its `type` tag.

    Dicts load into the class their `type` names; instances pass through.
    Dumps keep the subclass's own fields.

    Example:
        StepRef = ref(Step)

        class Sequence(Step):
            type: t.Literal["sequence"] = "sequence"
            steps: list[StepRef]
    """

    def dispatch(value: t.Any, handler: t.Callable[[t.Any], t.Any]) -> t.Any:
        if isinstance(value, root):
            return value
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            try:
                cls = root.resolve(value["type"])
            except LookupError as e:
                raise ValueError(str(e)) from None
            return cls.model_validate(value)
        return handler(value)

    return t.Annotated[root, WrapValidator(dispatch), SerializeAsAny()]
