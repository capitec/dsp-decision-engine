import typing as t
from abc import ABC, abstractmethod
from pydantic import BaseModel

from decikit import PolarsSchema

from .core import (
    DataSourceProtocol,
    DataSourceList,
    DataSourceInfo,
    DataSourceTablesResponse,
    DataSourceSearchResponse,
    PaginationInfo,
    DataSourceCapabilities,
    TableDetailResponse,
    TablePreviewResponse,
    TablePreviewData,
    TableInfo,
    DataSourceError,
    DataSourceNotFoundError,
    TableNotFoundError,
)
from dsp_decikit.cache.keys import CacheKeys
from .s3 import S3DataSource
from .local import LocalDataSource
from .settings import DataSourcesSettings


class DataSourceHandler(ABC):
    """Abstract base class for data source handlers"""

    @abstractmethod
    def create_datasource(self, config: BaseModel) -> DataSourceProtocol:
        """Create a data source instance from configuration"""
        pass


class DataSourceManager:
    """Manager for handling multiple configured data sources"""

    def __init__(self, settings: DataSourcesSettings):
        self.settings = settings
        self._datasources: t.Dict[str, DataSourceProtocol] = {}
        self._cache = None  # Lazy loaded to avoid circular imports

    @property
    def cache(self):
        """Lazy-loaded cache to avoid circular imports"""
        if self._cache is None:
            from dsp_decikit.settings import settings

            self._cache = settings.cache.create()
        return self._cache

    def get_datasource(self, name: str) -> DataSourceProtocol:
        """Get a data source by name, creating it if necessary"""
        if name not in self._datasources:
            config = self.settings.sources.get(name)
            if not config:
                raise DataSourceNotFoundError(f"Data source '{name}' is not configured")
            self._datasources[name] = config.create()
        return self._datasources[name]

    def list_datasources(self) -> DataSourceList:
        """List all configured data sources with their metadata"""
        datasources = []

        for name, config in self.settings.get_enabled_datasources().items():
            datasource_info = DataSourceInfo(
                name=name,
                provider=config.provider,
                display_name=config.display_name or name.replace("_", " ").title(),
                description=config.description,
                icon=config.icon,
                capabilities=config.capabilities(),
            )
            datasources.append(datasource_info)

        return DataSourceList(datasources=datasources)

    def get_datasources_by_capability(self, capability: str) -> DataSourceList:
        """Get data sources that support a specific capability"""
        all_sources = self.list_datasources()
        filtered_sources = [
            ds
            for ds in all_sources.datasources
            if DataSourceCapabilities(capability) in ds.capabilities
        ]
        return DataSourceList(datasources=filtered_sources)

    async def get_datasource_tables(
        self, name: str, page: int = 1, page_size: int = 50
    ) -> DataSourceTablesResponse:
        """Get tables from a specific data source"""
        datasource = self.get_datasource(name)
        response = await datasource.list_tables(page=page, page_size=page_size)

        return DataSourceTablesResponse(
            datasource=name,
            tables=response.items,
            pagination=PaginationInfo(
                page=response.page,
                page_size=response.page_size,
                total_count=response.total_count,
                has_next=response.has_next,
                has_previous=response.has_previous,
            ),
        )

    async def search_datasource_tables(
        self, name: str, query: str, page: int = 1, page_size: int = 50
    ) -> DataSourceSearchResponse:
        """Search tables in a specific data source"""
        datasource = self.get_datasource(name)
        response = await datasource.search_tables(
            query=query, page=page, page_size=page_size
        )

        return DataSourceSearchResponse(
            datasource=name,
            query=query,
            tables=response.items,
            pagination=PaginationInfo(
                page=response.page,
                page_size=response.page_size,
                total_count=response.total_count,
                has_next=response.has_next,
                has_previous=response.has_previous,
            ),
        )

    async def get_table_info(
        self, datasource_name: str, table_name: str
    ) -> TableDetailResponse:
        """Get detailed information about a specific table"""
        datasource = self.get_datasource(datasource_name)

        # Get table info and schema
        table_info = await datasource.get_table_info(table_name)
        schema = await datasource.get_columns(table_name)

        return TableDetailResponse(
            datasource=datasource_name,
            table=table_name,
            info=table_info,
            table_schema=schema,
        )

    async def preview_table_data(
        self, datasource_name: str, table_name: str, limit: int = 100
    ) -> TablePreviewResponse:
        """Get a preview of table data"""
        # Get datasource config to respect its limits and cache settings
        datasource_config = self.settings.sources.get(datasource_name)
        if not datasource_config:
            raise DataSourceNotFoundError(
                f"Data source '{datasource_name}' is not configured"
            )

        # Check cache first (if caching is enabled)
        cache_key = CacheKeys.datasource_preview(datasource_name, table_name, limit)
        if datasource_config.cache_ttl_seconds > 0:
            try:
                cached_preview = await self.cache.get(cache_key)
                return TablePreviewResponse.model_validate(cached_preview)
            except KeyError:
                pass  # Not in cache, proceed to fetch
            except Exception:
                # Bad cache data (validation failed) - delete and regenerate
                try:
                    await self.cache.delete(cache_key)
                except Exception:
                    pass  # Ignore cache delete errors

        datasource = self.get_datasource(datasource_name)

        try:
            # Use faster sampling method if available, otherwise fallback to regular method

            lazy_frame = await datasource.to_lazyframe(table_name)
            preview_df = lazy_frame.limit(datasource_config.max_preview_rows).collect()

            # Convert to structured data format with JSON serialization
            def make_json_serializable(value):
                """Convert any value to JSON serializable format"""
                if value is None:
                    return None
                elif hasattr(value, "isoformat"):  # datetime, date, time objects
                    return value.isoformat()
                elif isinstance(value, (bytes, bytearray)):
                    return value.decode("utf-8", errors="replace")
                elif hasattr(value, "item"):  # numpy types
                    return value.item()
                elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    return list(value)
                elif hasattr(value, "__dict__"):  # Complex objects
                    return str(value)
                else:
                    try:
                        # Test JSON serialization
                        import json

                        json.dumps(value)
                        return value
                    except (TypeError, ValueError):
                        return str(value)

            serializable_rows = []
            for row in preview_df.rows():
                serializable_row = [make_json_serializable(value) for value in row]
                serializable_rows.append(serializable_row)

            data = TablePreviewData(
                columns=preview_df.columns,
                rows=serializable_rows,
                row_count=len(serializable_rows),
                total_columns=len(preview_df.columns),
            )

            response = TablePreviewResponse(
                datasource=datasource_name, table=table_name, limit=limit, data=data
            )

            # Cache using datasource-specific TTL (if caching is enabled)
            if datasource_config.cache_ttl_seconds > 0:
                await self.cache.put(
                    cache_key,
                    response.model_dump(),
                    expire_seconds=datasource_config.cache_ttl_seconds,
                )
            return response

        except (DataSourceNotFoundError, TableNotFoundError):
            # Re-raise "not found" errors as-is
            raise
        except Exception as e:
            # Wrap other errors but don't convert validation errors
            raise DataSourceError(f"Failed to preview table data: {e}")
