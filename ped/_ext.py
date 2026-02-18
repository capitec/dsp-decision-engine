import typing as t
from pydantic import create_model, RootModel, Field
from warnings import warn

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
        root=("RootType", ...)  # The ... indicates required field
    )
    
    # State for tracking registered types
    _union_type = None
    _base_root_field = ExtendableModel.model_fields["root"]
    
    # We could consider injecting register_provider into __init_subclass__ but i like the verbosity
    # and im not sure im a fan of the side effects of importing a class causing changes to the model
    def register_provider(provider_class: t.Type):
        nonlocal _union_type, _base_root_field
        assert issubclass(provider_class, base_class), f"Provider must be a subclass of {base_class.__name__}"
        
        # Note: Python is smart enough to optimize unions, so this won't create nested unions
        # So t.Union[A] = A and t.Union[A, t.Union[B, C]] = t.Union[A, B, C]
        if _union_type is None:
            _union_type = provider_class
        else:
            _union_type = t.Union[_union_type, provider_class]
        # Update the model's root type annotation
        # Pydantic hardcodes this on the first update to not be a reference to RootType
        # Due to that we need to reset it to the root type before rebuilding the model
        ExtendableModel.model_fields["root"] = _base_root_field
        was_rebuilt = ExtendableModel.model_rebuild(
            force=True, 
            _types_namespace={"RootType": t.Annotated[_union_type, Field(discriminator=discriminator_field)]}
        )
        if was_rebuilt != True:
            warn(f"{provider_class.__name__} was not properly rebuilt into {ExtendableModel.__name__}")
    
    return ExtendableModel, register_provider
