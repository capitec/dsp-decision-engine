import typing as t
from pydantic import BaseModel, model_serializer, FieldSerializationInfo, SerializerFunctionWrapHandler, model_validator, ConfigDict, field_validator, Field, PrivateAttr
from dataclasses import field
from typing import TYPE_CHECKING
from decider.modules.core import BaseModule
from decider.modules import GraphModule


class ConfigModule(BaseModule):
    model_config = ConfigDict(extra='allow')
    type: t.Literal["config"]
    inner_module_type: str
    config_overrides: t.List[t.Tuple[str,...]]

    _ModuleType: t.Type[BaseModule]
    _module_config: BaseModel = PrivateAttr(default=None)

    @property
    def config(self) -> "ConfigSelector":
        return ConfigSelector(self)

    def _config_model_class(self) -> BaseModel:
        ConfigModel = self._ModuleType
        # 1. start with _ModuleType as a base model
        # loop through config_overrides and replace the type at the end of the override the type to be a parameter type
        return ConfigModel

    def _rebuild_module_config(self):
        ConfigClass = self._config_model_class()
        if self.model_config is not None:
            model_config_raw = self.model_config.model_dump()
        else:
            model_config_raw = self.model_extra
        self._module_config = ConfigClass.model_validate(model_config_raw)

    @field_validator()
    #...
    def ensure_no_override_overlap(config_overrides):
        ...
        # TODO ensure that there is no path overlap
        # ie there cannot be ("param1", "param2", "0", "x") and ("param1", "param2") as the one is a subpath of the other 

    @model_validator(mode='after')
    def initialise_internal_components(self):
        self._ModuleType = GraphModule.root # <- TODO use inner_module_type to fetch the Class from the union
        self._rebuild_module_config()


    @model_serializer(mode='wrap')
    def serialize_model(
        self, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> dict[str, object]:
        self._rebuild_module_config() # Just in-case
        serialized = handler(self)
        return {
            **self._module_config.model_dump(mode=info.mode),
            "type": serialized["type"],
            "inner_module_type": serialized["inner_module_type"],
            "config_overrides": serialized["config_overrides"],
        }