import inspect
import typing as t
from abc import ABC
from pydantic import create_model, RootModel, BaseModel, Field, model_validator
from warnings import warn


class TypeDiscriminatedBaseModule(BaseModel, ABC):
    type: str 

    _CLASS_TYPE_IDENTIFIER: t.ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        """
        We are basically using the below to ensure:
        1. the class implements a type: Literal['value'] so we can use that as a discriminator for the union of all implementations of this class
        2. We dont want there to be type: Literal['value'] = 'value' on the class because we making use of pydantic.model_dump(exclude_defaults=True) to exclude the type field when saving out modules, and if there is a default value then it will not be included in the dumdecider dict which breaks loading it back in.
        3. We want to store what the value of Literal is so we can automatically initialise it when we construct the model Model() rather than needing Model(type='value') every time
        """
        super().__init_subclass__(**kwargs)

        # Skip abstract classes. as this will be used as a base for multiple implementations.
        if inspect.isabstract(cls):
            return

        # Ensure `type` declared
        if "type" not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' annotation")

        annotation = cls.__annotations__["type"]

        if t.get_origin(annotation) is not t.Literal:
            raise TypeError(
                f"{cls.__name__}.type must be typing.Literal[...]"
            )

        literal_values = t.get_args(annotation)

        if len(literal_values) != 1:
            raise TypeError(
                f"{cls.__name__}.type must be a single-value Literal"
            )

        if "type" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}.type must not define a default value"
            )

        cls._CLASS_TYPE_IDENTIFIER = literal_values[0]

    @model_validator(mode="before")
    @classmethod
    def auto_set_type(cls, values):
        if isinstance(values, dict) and not inspect.isabstract(cls):
            values.setdefault("type", cls._CLASS_TYPE_IDENTIFIER)
        return values


_TExtenableRootType = t.TypeVar("_TExtenableRootType")

class TExtendableModel(RootModel[_TExtenableRootType], t.Generic[_TExtenableRootType]):
    root: _TExtenableRootType


def create_extendable_model(
    base_class: t.Type, 
    discriminator_field: str = "type", 
    model_name: str = "ExtendableModel"
) -> t.Tuple[t.Type[TExtendableModel], t.Callable[[t.Type], None]]:
    """
    Creates an extendable model pattern that allows external packages to register
    new types without creating hard dependencies.
    
    Returns a tuple of (Model class, register function).
    """
    
    ExtendableModel = create_model(
        model_name,
        __base__=RootModel,
        root=("RootType", ...)
    )
    
    _registered: t.Dict[str, t.Type] = {}

    def _rebuild():
        classes = list(_registered.values())
        if not classes:
            return
        union = classes[0] if len(classes) == 1 else t.Union[tuple(classes)]
        annotated = t.Annotated[union, Field(discriminator=discriminator_field)]
        ExtendableModel.__annotations__["root"] = annotated
        ExtendableModel.model_fields["root"].annotation = annotated
        was_rebuilt = ExtendableModel.model_rebuild(
            force=True,
            _types_namespace={"RootType": annotated},
        )
        if was_rebuilt is not True:
            warn(f"model_rebuild did not return True for {ExtendableModel.__name__}")

    def register_provider(provider_class: t.Type):
        assert issubclass(provider_class, base_class), f"Provider must be a subclass of {base_class.__name__}"
        type_id = getattr(provider_class, "_CLASS_TYPE_IDENTIFIER", None) or provider_class.__name__
        _registered[type_id] = provider_class
        _rebuild()
    
    return ExtendableModel, register_provider
