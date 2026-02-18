"""Flow configuration parser and model definitions"""

import typing as t
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, PrivateAttr
from .param.sources import ParameterSource
from .param.context import config_source_provider
from .param.param_source import ParameterSourceProvider, VersionMismatchStrategy
from omegaconf import OmegaConf, DictConfig


class FlowMetadata(BaseModel):
    """Flow metadata information"""
    model_config = ConfigDict(extra="allow")
    name: str = Field(description="Flow name")
    description: str = Field(description="Flow description")
    author: str
    license: str = Field(default="MIT")
    version: str = Field(default="1.0")

class ParameterSourceConfig(BaseModel):
    """Configuration for a parameter source"""
    kwargs: t.Dict[str, t.Any] = Field(default_factory=dict, description="Additional keyword arguments for the parameter source")
    version_missmatch_strategy: VersionMismatchStrategy = Field(
        default=VersionMismatchStrategy.ERROR,
        description="Strategy to handle version mismatches between requested and actual parameter versions."
    )

class FlowConfiguration(BaseModel):
    """Main flow configuration model"""
    metadata: FlowMetadata
    
    parameter_sources: t.Dict[str, ParameterSource] = Field(
        default_factory=dict,
        description="Named parameter sources configuration"
    )
    
    # Note these are dictionaries still so that we can use omegaconf to update before we 
    # parse them into modules. This helps make things dynamic
    modules: t.Dict[str, t.Dict[str, t.Any]] = Field(
        description="Module definitions and configurations."
    )
    parameter_source_config: ParameterSourceConfig = Field(
        default_factory=ParameterSourceConfig,
        description="Configuration for parameter sources, including version mismatch strategy."
    )

    _provider: "ParameterSourceProvider" = PrivateAttr()
    _modules_config: DictConfig = PrivateAttr(default=None)


    @model_validator(mode='before')
    @classmethod
    def enable_omegaconf(cls, values):
        if not isinstance(values, dict):
            return values
        try:
            with config_source_provider(
                ParameterSourceProvider(
                    sources={}, # TODO can add some default sources
                ), 
            ):
                return OmegaConf.to_object(OmegaConf.create(values))
        except Exception:
            # If we fail to parse with omegaconf just return the original values and let pydantic handle the error
            return values
        

    @field_validator('modules')
    def validate_modules_not_empty(cls, v):
        if not v:
            raise ValueError("At least one module must be defined")
        return v
    
    @model_validator(mode='after')
    def build_provider(self):
        # TODO wondering if the cache layer on the sources is needed. if they are this would be the place
        # to call source.get_cached_source
        # If we remove the cache we can probably just build this object each time might be cleaner
        provider = ParameterSourceProvider(sources=self.parameter_sources)
        self._provider = provider
        return self
    

    def get_parameterized_modules(
        self,
        request: t.Any = None,
        requested_versions: t.Dict[str, t.Any] | t.Any | None = None,
        rebuild_modules: bool = False,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        if requested_versions is not None and not isinstance(requested_versions, dict):
            requested_versions = {source_name: requested_versions for source_name in self.parameter_sources.keys()}
        if requested_versions is None:
            requested_versions = {}
        if request is None:
            request = {}
        if self._modules_config is None or rebuild_modules:
            self._modules_config = OmegaConf.create(self.modules)
        with config_source_provider(
            ParameterSourceProvider(
                sources=self.parameter_sources,
                request=request, 
                requested_versions=requested_versions,
                version_mismatch_strategy=self.parameter_source_config.version_missmatch_strategy,
                kwargs=self.parameter_source_config.kwargs,
                # Note we dont inject current version as that is more useful to determine if we need to reload the config
                # When we are getting config we just want to get the config at a version
            ), 
        ):
            return OmegaConf.to_object(self._modules_config)
