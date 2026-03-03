import typing as t
import json
import importlib
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator, model_serializer, SerializationInfo

from .namespaced import NamespacedModule
from ped.param.sources.core import DictVersionedSource
from ped.param.param_source import ParameterSourceProvider
from ped.param.context import config_source_provider
import ped.param  # noqa: F401 – ensures the "ped.param" OmegaConf resolver is registered


@lru_cache(maxsize=128)
def load_external_config(config_path: str) -> dict:
    """Load external module configuration from a JSON file (cached by path)."""
    with open(config_path, "r") as f:
        return json.load(f)



class ExternalReference(BaseModel):
    # Doing a bit of a hack here because we use this as the serialized model for below
    # and we use the model validate to inject the model.
    type: t.Literal["external"] = "external"
    module_name: str = Field(description="Name of the external module being referenced")
    config_path: str = Field(
        description="Path to the JSON file containing the module definition. Must be relative to the module.", 
        default="module.json"
    )
    parameters: t.Dict[str, t.Any] = Field(
        default_factory=dict,
        description=(
            "Values injected as the 'params' OmegaConf source when resolving "
            "the loaded config. Use ${ped.param:params,<key>} placeholders in "
            "the saved JSON to reference these."
        ),
    )

    def load(self) -> "t.Dict[str, t.Any]":
        """Load the external module definition from the JSON file and return an ExternalModule instance."""
        module = importlib.import_module(self.module_name)
        # Get the config path relative to the module path
        module_path = Path(module.__file__).parent
        config_path = module_path / self.config_path


        loaded_config: dict = dict(load_external_config(config_path))

        # Resolve OmegaConf interpolations (${ped.param:params,<key>}) using
        # the provided parameters as the "params" source.
        with config_source_provider(
            ParameterSourceProvider(
                sources={"ext": DictVersionedSource(values=self.parameters)},
                inputs={},
            )
        ):
            loaded_config = OmegaConf.to_object(OmegaConf.create(loaded_config))
        # Store a reference to this model so we can serialize it later
        return loaded_config | {"ref": self}

class ExternalModule(NamespacedModule):
    """
    """

    type: t.Literal["external"]
    ref: ExternalReference = Field(description="Reference to the external module definition")

    @model_validator(mode="before")
    @classmethod
    def load_config(cls, values: t.Union[dict, "ExternalReference", "ExternalModule"]) -> dict:
        """Load the external JSON and merge it with any explicitly passed values."""
        if isinstance(values, ExternalReference):
            return values.load()
        if not isinstance(values, dict):
            return values
        # Just a quick check to see if this is a Ref or a direct External Module
        # Ideally it should always be a ref when loading
        if "module_name" not in values:
            return values
        
        return ExternalReference.model_validate(values).load()

    @model_serializer(mode="wrap")
    def serialize_model(self, handler, info: SerializationInfo):
        """
        We use this so that when the flow is saved it only saves the ref to the module and not the full module definition.
        This is unless we do a dump with 'full_model_dump' in the context, in which case we include the full module definition. 
        This is used when exporting the module to ensure we save the full definition to the JSON file.
        We can maybe consider a more unique name here but i think this is intuitive.
        """
        if info.context and info.context.get("full_model_dump"):
            return handler(self)

        return self.ref.model_dump()

    def export(self):
        from ped.initialization import ext_settings

        # Get the path to save the module
        module_path = Path(ext_settings.extension_path) / self.ref.module_name
        config_path = module_path / self.ref.config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the config as JSON
        with open(config_path, "w") as f:
            json.dump(self.model_dump(
                exclude={"ref"}, 
                context={"full_model_dump": True}
            ), f, indent=4)
        # Create an init file if none exists to make it a package
        init_path = module_path / "__init__.py"
        init_path.touch(exist_ok=True)
        print(f"External module exported to {config_path}. If this module makes use of any extensions please ensure you transfer the code into this path.")
