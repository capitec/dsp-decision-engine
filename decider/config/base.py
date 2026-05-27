import typing as t
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_serializer, model_validator, SerializerFunctionWrapHandler, SerializationInfo
from .versioned import Version, get_current_versioned_config

T = t.TypeVar("T", bound=BaseModel)

DUMP_TRIGGER_KEY = "dump_to_versioned_config"


class BaseConfig(BaseModel, t.Generic[T]):
    model_config = ConfigDict(extra="allow")
    config_key: str
    _constructed_model: T
    _loaded_version: Version
    _MODEL_CLASS: t.ClassVar[t.Type[T]]

    @model_validator(mode="after")
    def _construct_model(self) -> "t.Self":
        """Construct the internal model from the config dict."""
        constructed_model = self.model_extra.pop("_constructed_model", None)
        if constructed_model is not None:
            assert isinstance(constructed_model, self._MODEL_CLASS), (
                f"_constructed_model must be an instance of {self._MODEL_CLASS.__name__}, got {type(constructed_model).__name__}"
            )
            loaded_version = self.model_extra.pop("_loaded_version", Version(-float('inf'),0,0))
            assert isinstance(loaded_version, Version), (
                f"_loaded_version must be an instance of Version or None, got {type(loaded_version).__name__}"
            )

            self._constructed_model = constructed_model
            self._loaded_version = loaded_version
            return self

        self.reload(force=True)
        return self
    
    @classmethod
    def from_model(cls, model: T, config_key:str) -> "BaseConfig[T]":
        """Create a BaseConfig instance from a constructed model. This is the preferred way to create a new config, as it ensures the internal state is consistent."""
        return cls(
            config_key=config_key, 
            _constructed_model=model,
        )
    
    def reload(self, force=False) -> None:
        """Reload the internal model from the current versioned config. Call this if you know the underlying config has changed and you want to refresh the model."""
        versioned_config = get_current_versioned_config()
        if versioned_config is None:
            raise RuntimeError("No versioned config found in context. Make sure to use current_version_context() when accessing the config.")
        if not force and versioned_config.version == self._loaded_version:
            return  # No need to reload if the version hasn't changed
        config_data = versioned_config.config.get(self.config_key)
        if config_data is None:
            raise RuntimeError(f"Config key {self.config_key!r} not found in versioned config.")
        self._constructed_model = self._MODEL_CLASS.model_validate(config_data)
        self._loaded_version = versioned_config.version

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict:
        result = handler(self)
        ctx = info.context
        if ctx.get(DUMP_TRIGGER_KEY, False):
            versioned_config = get_current_versioned_config()
            if versioned_config is None:
                raise RuntimeError("No versioned config found in context. Make sure to use current_version_context() when accessing the config.")
            # Update the versioned config with the current model's data
            versioned_config.config[self.config_key] = self._constructed_model.model_dump()
        return result