import typing as t
from pydantic import RootModel, Field

_TExtenableRootType = t.TypeVar("_TExtenableRootType")

class TExtendableModel(RootModel[_TExtenableRootType], t.Generic[_TExtenableRootType]):
    root: _TExtenableRootType


def create_extendable_model(base_class: t.Type, discriminator_field: str = "type", model_name: str = "ExtendableModel"):
    """
    Creates an extendable model pattern that allows external packages to register
    new types without creating hard dependencies.
    
    Returns a tuple of (Model class, register function).
    """
    
    class ExtendableModel(RootModel):
        # Make use of pydantic forward ref
        root: "_annotated_type" # pyright: ignore[reportInvalidTypeForm]

    # Set the class name dynamically
    ExtendableModel.__name__ = model_name
    ExtendableModel.__qualname__ = model_name
    
    # State for tracking registered types
    _union_type = None
    _annotated_type = None
    
    # We could consider injecting register_provider into __init_subclass__ but i like the verbosity
    # and im not sure im a fan of the side effects of importing a class causing changes to the model
    def register_provider(provider_class: t.Type):
        nonlocal _union_type, _annotated_type
        assert issubclass(provider_class, base_class), f"Provider must be a subclass of {base_class.__name__}"
        
        # Note: Python is smart enough to optimize unions, so this won't create nested unions
        # So t.Union[A] = A and t.Union[A, t.Union[B, C]] = t.Union[A, B, C]
        _union_type = t.Union[_union_type, provider_class]
        # Update the model's root type annotation
        _annotated_type = t.Annotated[_union_type, Field(discriminator=discriminator_field)]
        ExtendableModel.model_rebuild(force=True)
    
    return ExtendableModel, register_provider
