"""Flow configuration parser and model definitions"""

import typing as t
import asyncio
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, PrivateAttr
from functools import cached_property
from .param.sources import ParameterSource
from .param.context import config_source_provider
from .param.param_source import ParameterSourceProvider, VersionedSource
from .param.types import TVersionType
from ped.types import TInputType
from .graph import GraphBuilder, BaseGraph
from .cache import CacheProvider, lru
from .modules import ConstructedGraphModules
from omegaconf import OmegaConf, DictConfig


class FlowMetadata(BaseModel):
    """Flow metadata information"""
    model_config = ConfigDict(extra="allow")
    name: str = Field(description="Flow name")
    description: str = Field(description="Flow description")
    author: str
    license: str = Field(default="MIT")
    version: str = Field(default="1.0")


class FlowConfiguration(BaseModel):
    """Main flow configuration model"""
    metadata: FlowMetadata
    
    parameter_sources: t.Dict[str, ParameterSource] = Field(
        default_factory=dict,
        description="Named parameter sources configuration"
    )
    
    # Note these are dictionaries still so that we can use omegaconf to update before we 
    # parse them into modules. This helps make things dynamic
    modules: t.List[t.Dict[str, t.Any]] = Field(
        description="Module definitions and configurations."
    )

    graph_builder: GraphBuilder = Field(
        default_factory=lambda: GraphBuilder(type="hamilton"),
        description="Graph builder instance to use for building the computational graph."
    )

    cache: CacheProvider = Field(
        default_factory=lambda: CacheProvider(type="lru"),
        description="Caching configuration for the builder"
    )
    outputs: t.List[str] = Field(
        default_factory=list,
        description="List of output node names for the graph."
    )

    _modules_config: DictConfig = PrivateAttr(default=None)


    @cached_property
    def _source_names(self) -> t.List[str]:
        return list(self.parameter_sources.keys())


    @model_validator(mode='before')
    @classmethod
    def enable_omegaconf(cls, values):
        if not isinstance(values, dict):
            return values
        try:
            return OmegaConf.to_object(OmegaConf.create(values))
        except Exception:
            # If we fail to parse with omegaconf just return the original values and let pydantic handle the error
            return values
        

    @field_validator('modules')
    def validate_modules_not_empty(cls, v):
        if not v:
            raise ValueError("At least one module must be defined")
        return v
    

    def get_parameterized_modules_from_sources(
        self,
        inputs: TInputType,
        versioned_sources: t.Dict[str, VersionedSource],
        rebuild_modules: bool = False,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        if self._modules_config is None or rebuild_modules:
            self._modules_config = OmegaConf.create(self.modules)
        with config_source_provider(
            ParameterSourceProvider(
                sources=versioned_sources,
                inputs=inputs, 
            ), 
        ):
            return OmegaConf.to_object(self._modules_config)
    
    async def get_source_versions(
        self, 
        inputs: TInputType
    ) -> t.Tuple[TVersionType]:
        # TODO evaluate if we should try send in the cached versions here
        # I dont think it makes sense because we should just determine the versions based on the inputs 
        # sending cached versions makes it very complicate as there could be multiple cached versions that the sources would have to consider.
        versions = await asyncio.gather(*[
            self.parameter_sources[source_name].root.get_version(inputs=inputs)
            for source_name in self._source_names
        ])
        return tuple(versions)
    
    async def get_parameterized_module_config(
        self,
        inputs: TInputType,
        source_versions: t.Optional[t.Tuple[TVersionType]] = None,
        rebuild_modules: bool = False,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        if source_versions is None:
            source_versions = await self.get_source_versions(inputs=inputs)
        versioned_sources = await asyncio.gather(*[
            self.parameter_sources[source_name].root.get_versioned_source(inputs=inputs, version=version)
            for source_name, version in zip(self._source_names, source_versions)
        ])
        versioned_sources_dict = {
            source_name: versioned_source
            for source_name, versioned_source in zip(self._source_names, versioned_sources)
        }
        return self.get_parameterized_modules_from_sources(
            inputs=inputs,
            versioned_sources=versioned_sources_dict,
            rebuild_modules=rebuild_modules,
        )
    

    async def get_parameterized_modules(
        self,
        inputs: TInputType,
        source_versions: t.Optional[t.Tuple[TVersionType]] = None,
        rebuild_modules: bool = False,
    ) -> "ConstructedGraphModules":
        parameterized_module_config = await self.get_parameterized_module_config(
            inputs=inputs, 
            source_versions=source_versions, 
            rebuild_modules=rebuild_modules,
        )
        # We can add some additional information to the module config here if needed before we build the graph
        return ConstructedGraphModules.model_validate({"root": parameterized_module_config})
    
    async def build_graph(
        self,
        inputs: t.Any = None,
        source_versions: t.Optional[t.Tuple[TVersionType]] = None,
        rebuild_modules: bool = False,
    ) -> "BaseGraph":
        """Build the graph based on the current configuration and provided builder config."""
        parameterized_modules = await self.get_parameterized_modules(
            inputs=inputs, 
            source_versions=source_versions, 
            rebuild_modules=rebuild_modules,
        )
        # We can add some additional information to the builder config here if needed before we build the graph
        return self.graph_builder.root.build_graph(
            parameterized_modules,
            output_nodes=self.outputs,
        )

    async def get_graph(
        self,
        inputs: TInputType = None,
        rebuild_modules: bool = False,
    ) -> "BaseGraph":
        """Get the graph from cache if available and valid, otherwise build a new graph."""
        source_versions = await self.get_source_versions(inputs=inputs)
        if rebuild_modules or self.cache.root.has(source_versions):
            return self.cache.root.get(source_versions)
        
        async with self.cache.root.lock(source_versions):
            # Check again in lock because someone else might've added it
            if rebuild_modules or self.cache.root.has(source_versions):
                return self.cache.root.get(source_versions)
            # TODO the question is do we build this in a lock so multiple people trying to access the graph at once might have to wait
            # Or do we build it outside of the lock but then run the extra computation of building the graph multiple times
            # Im thinking with this design the cache can decide because it can do nothing in the lock code and this will work
            graph = await self.build_graph(
                inputs=inputs, 
                source_versions=source_versions, 
                rebuild_modules=rebuild_modules,
            )
            await self.cache.root.put(source_versions, graph)
        return graph

    # def get_graph(
    #     self,
    #     inputs: TInputType = None,
    #     rebuild_modules: bool = False,
    # ) -> "BaseGraph":
    #     """Get the graph from cache if available and valid, otherwise build a new graph."""
    #     cached_versions: t.Dict[str, t.Set[TVersionType]] = self.cache.get_cached_versions()
    #     requested_versions = self.parameter_versions.infer_versions(inputs)
        
    #     if requested_versions is not None:
    #         version_composite_key = tuple(
    #             requested_versions.get(source_name)  
    #             for source_name in self._source_names
    #         )
    #         if None in version_composite_key:
    #             # We cannot make use of the quick cache here as there is a version that always wants to make use of 
    #             # the latest version so we will have to check each source individually to see if it needs to be refreshed
    #             for source_name, requested_version in zip(self._source_names, version_composite_key):
    #                 if self.parameter_sources[source_name].requires_refresh(
    #                     inputs=inputs,
    #                     curr_version=cached_versions.get(source_name),
    #                     requested_version=requested_version,
    #                     **self.parameter_source_config.kwargs,
    #                 ):
    #                     break
    #         elif version_composite_key in cached_versions:
    #             return self.cache.get_graph(version_composite_key)
    #         else:
    #             graph = self.build_graph(
    #                 inputs=inputs, 
    #                 requested_versions=requested_versions, 
    #                 rebuild_modules=rebuild_modules,
    #             )
    #             self.cache.store_graph(version_composite_key, graph)
    #             return graph
        
        