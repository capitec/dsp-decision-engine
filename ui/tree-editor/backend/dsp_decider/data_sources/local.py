import typing as t
from pathlib import Path
import asyncio

import polars as pl
from pydantic import BaseModel, Field, field_validator

from .core import (
    DataSourceCapabilities,
    DataSourceProtocol,
    DataSourceConfig,
    PaginatedResponse,
    TableInfo,
    DataSourceError,
    TableNotFoundError,
    ConnectionError,
)
from dsp_decider.cache.keys import CacheKeys
from decider import PolarsSchema


class LocalDataSourceConfig(DataSourceConfig):
    """Configuration for local file system data source"""

    provider: t.Literal["local"] = "local"
    _SUPPORTED_CAPABILITIES: t.ClassVar[t.Set[DataSourceCapabilities]] = {
        DataSourceCapabilities.DISCOVERY,
        DataSourceCapabilities.SEARCH,
        DataSourceCapabilities.PREVIEW,
        DataSourceCapabilities.UPLOAD,
    }

    # Local directory path
    directory_path: str = Field(
        ..., description="Local directory path containing table subdirectories"
    )

    @field_validator("directory_path")
    @classmethod
    def validate_directory_path(cls, v: str) -> str:
        """Validate that directory_path is a valid directory"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return str(path.absolute())

    def create(self) -> "LocalDataSource":
        """Create a LocalDataSource instance from this configuration"""
        return LocalDataSource(self)


class LocalDataSource(DataSourceProtocol):
    """Local file system data source implementation for parquet files organized as /table_name/data_chunk_*.parquet"""

    def __init__(self, config: LocalDataSourceConfig):
        self.config = config
        self.base_path = Path(config.directory_path)
        self._cache = None  # Lazy loaded to avoid circular imports

    @property
    def cache(self):
        """Lazy-loaded cache to avoid circular imports"""
        if self._cache is None:
            from dsp_decider.settings import settings

            self._cache = settings.cache.create()
        return self._cache

    async def _run_sync_in_thread(self, func: t.Callable, *args, **kwargs):
        """Helper to run synchronous file operations in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _list_table_directories(self) -> t.List[str]:
        """List all table directories (folders) in the base directory"""
        try:

            def _list_dirs():
                table_names = []
                for item in self.base_path.iterdir():
                    if item.is_dir():
                        table_names.append(item.name)
                return table_names

            return await self._run_sync_in_thread(_list_dirs)

        except OSError as e:
            raise ConnectionError(f"Failed to list directories: {e}")

    async def list_tables(
        self, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """List all available tables with pagination"""
        all_tables = await self._list_table_directories()
        all_tables.sort()  # Sort alphabetically

        total_count = len(all_tables)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        items = all_tables[start_idx:end_idx]

        return PaginatedResponse(
            items=items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_count,
            has_previous=page > 1,
        )

    async def search_tables(
        self, query: str, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """Search for tables matching the query string"""
        all_tables = await self._list_table_directories()

        # Simple case-insensitive substring search
        query_lower = query.lower()
        matching_tables = [
            table for table in all_tables if query_lower in table.lower()
        ]
        matching_tables.sort()

        total_count = len(matching_tables)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        items = matching_tables[start_idx:end_idx]

        return PaginatedResponse(
            items=items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_count,
            has_previous=page > 1,
        )

    async def _get_table_parquet_files(self, table_name: str) -> t.List[Path]:
        """Get all parquet files for a given table"""
        table_path = self.base_path / table_name

        if not table_path.exists():
            raise TableNotFoundError(f"Table directory not found: {table_name}")

        if not table_path.is_dir():
            raise DataSourceError(f"Table path is not a directory: {table_name}")

        try:

            def _list_parquet_files():
                parquet_files = list(table_path.glob("*.parquet"))
                if not parquet_files:
                    raise TableNotFoundError(
                        f"No parquet files found for table '{table_name}'"
                    )
                return parquet_files

            return await self._run_sync_in_thread(_list_parquet_files)

        except OSError as e:
            raise ConnectionError(
                f"Failed to list parquet files for table '{table_name}': {e}"
            )

    async def to_lazyframe(self, table_name: str) -> pl.LazyFrame:
        """Get a Polars LazyFrame view of the specified table"""
        table_path = self.base_path / table_name

        if not table_path.exists():
            raise TableNotFoundError(f"Table directory not found: {table_name}")

        # Use glob pattern for better performance - Polars can handle this natively
        glob_pattern = str(table_path / "*.parquet")

        try:
            return pl.scan_parquet(glob_pattern)
        except Exception as e:
            raise DataSourceError(
                f"Failed to create LazyFrame for table '{table_name}': {e}"
            )

    async def get_columns(self, table_name: str) -> PolarsSchema:
        """Get the column names and types for a table as a PolarsSchema"""
        # Check cache first
        cache_key = CacheKeys.datasource_schema(str(self.base_path), table_name)
        try:
            cached_schema_dict = await self.cache.get(cache_key)
            return PolarsSchema.model_validate(cached_schema_dict)
        except KeyError:
            pass  # Not in cache, proceed to fetch
        except Exception:
            # Bad cache data (validation failed) - delete and regenerate
            try:
                await self.cache.delete(cache_key)
            except Exception:
                pass  # Ignore cache delete errors

        # Get the schema from the LazyFrame
        lazy_frame = await self.to_lazyframe(table_name)

        try:
            # Use collect_schema() to avoid performance warning and be more efficient
            pl_schema = lazy_frame.collect_schema()
            polars_schema = PolarsSchema.from_polars(pl_schema)

            # Cache using configurable TTL (schemas don't change often)
            if self.config.schema_cache_ttl_seconds > 0:
                await self.cache.put(
                    cache_key,
                    polars_schema.model_dump(),
                    expire_seconds=self.config.schema_cache_ttl_seconds,
                )
            return polars_schema

        except Exception as e:
            raise DataSourceError(f"Failed to get schema for table '{table_name}': {e}")

    async def get_table_info(self, table_name: str) -> TableInfo:
        """Get metadata information about a table"""
        parquet_files = await self._get_table_parquet_files(table_name)

        try:

            def _get_file_info():
                total_size = 0
                last_modified = None

                for file_path in parquet_files:
                    stat = file_path.stat()
                    total_size += stat.st_size
                    modified = stat.st_mtime

                    if last_modified is None or modified > last_modified:
                        last_modified = modified

                return total_size, last_modified

            total_size, last_modified = await self._run_sync_in_thread(_get_file_info)

            # Try to get row count from LazyFrame
            row_count = None
            try:
                lazy_frame = await self.to_lazyframe(table_name)
                # Note: This will actually scan the data to count rows
                # For very large tables, you might want to skip this or make it optional
                row_count = lazy_frame.select(pl.len()).collect().item()
            except Exception:
                # If we can't get row count, that's okay
                pass

            # Convert timestamp to ISO format
            from datetime import datetime

            last_modified_iso = (
                datetime.fromtimestamp(last_modified).isoformat()
                if last_modified
                else None
            )

            return TableInfo(
                name=table_name,
                row_count=row_count,
                size_bytes=total_size,
                last_modified=last_modified_iso,
            )

        except OSError as e:
            raise ConnectionError(f"Failed to get info for table '{table_name}': {e}")

    async def upload_file(self, filename: str, content: bytes) -> str:
        """Upload a file to local storage, converting to Parquet format first, and return the table name"""
        file_path = Path(filename)

        # Validate file extension - we support more formats now since we convert to Parquet
        allowed_extensions = [".csv", ".parquet", ".json", ".jsonl", ".xlsx"]
        if file_path.suffix.lower() not in allowed_extensions:
            raise DataSourceError(
                f"File extension {file_path.suffix} not allowed. Allowed: {', '.join(allowed_extensions)}"
            )

        # Use filename without extension as table name
        table_name = file_path.stem

        try:
            # Convert content to Parquet using Polars
            parquet_content = await self._convert_to_parquet(
                content, file_path.suffix.lower()
            )

            # Create table directory if it doesn't exist
            table_dir = self.base_path / table_name
            table_dir.mkdir(parents=True, exist_ok=True)

            # Write the parquet file: table_name/data.parquet
            output_path = table_dir / "data.parquet"

            def _write_file():
                output_path.write_bytes(parquet_content)

            await self._run_sync_in_thread(_write_file)
            return table_name

        except Exception as e:
            if isinstance(e, DataSourceError):
                raise
            raise DataSourceError(f"Failed to upload and convert file to Parquet: {e}")

    async def _convert_to_parquet(self, content: bytes, file_extension: str) -> bytes:
        """Convert uploaded file content to Parquet format using Polars"""
        from io import BytesIO

        def _do_conversion():
            try:
                content_buffer = BytesIO(content)

                # Create DataFrame based on file type
                if file_extension == ".csv":
                    df = pl.read_csv(content_buffer)
                elif file_extension == ".json":
                    df = pl.read_json(content_buffer)
                elif file_extension == ".jsonl":
                    df = pl.read_ndjson(content_buffer)
                elif file_extension == ".parquet":
                    # Already Parquet, just return the content
                    return content
                elif file_extension == ".xlsx":
                    # Polars can read Excel files from BytesIO directly
                    df = pl.read_excel(content_buffer)
                else:
                    raise DataSourceError(f"Unsupported file format: {file_extension}")

                # Convert DataFrame to Parquet bytes
                parquet_buffer = BytesIO()
                df.write_parquet(parquet_buffer, compression="snappy")
                return parquet_buffer.getvalue()

            except Exception as e:
                raise DataSourceError(
                    f"Failed to convert {file_extension} to Parquet: {e}"
                )

        return await self._run_sync_in_thread(_do_conversion)
