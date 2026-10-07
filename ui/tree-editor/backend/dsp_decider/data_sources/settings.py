import typing as t
from typing import Annotated
from pydantic import Field, BaseModel, BeforeValidator
from .datasource_registry import TDataSources
from .core import DataSourceCapabilities


class DataSourcesSettings(BaseModel):
    """Configuration for all data sources"""

    # Dictionary of named data source configurations
    sources: t.Dict[str, TDataSources] = Field(
        default_factory=dict, description="Named data source configurations"
    )

    # Global settings
    default_page_size: int = Field(
        default=50, description="Default page size for listings"
    )
    max_page_size: int = Field(
        default=500, description="Maximum page size for listings"
    )
    cache_ttl_seconds: int = Field(
        default=300, description="Cache TTL for data source metadata"
    )

    def get_enabled_datasources(
        self,
    ) -> t.Dict[
        str,
        TDataSources,
    ]:
        """Get only enabled data sources"""
        return {**self.sources}

    def get_datasources_by_capability(
        self, capability: DataSourceCapabilities
    ) -> t.Dict[
        str,
        TDataSources,
    ]:
        """Get data sources that support a specific capability"""
        enabled = self.get_enabled_datasources()
        return {
            name: config
            for name, config in enabled.items()
            if capability in config.capabilities()
        }
