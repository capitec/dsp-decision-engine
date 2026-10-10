import typing as t
from pathlib import Path
import asyncio
from datetime import datetime

import polars as pl
from pydantic import Field, field_validator

from .core import (
    DataSourceProtocol,
    DataSourceConfig,
    PaginatedResponse,
    TableInfo,
    DataSourceError,
    TableNotFoundError,
    ConnectionError,
)


class UploadDataSourceConfig(DataSourceConfig):
    """Configuration for upload-based data source"""

    storage_path: str = Field(..., description="Local storage path for uploaded files")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    allowed_extensions: t.List[str] = Field(
        default_factory=lambda: [".csv", ".parquet", ".json", ".jsonl"],
        description="Allowed file extensions",
    )

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        """Validate and create storage path if it doesn't exist"""
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @field_validator("allowed_extensions")
    @classmethod
    def validate_extensions(cls, v: t.List[str]) -> t.List[str]:
        """Ensure extensions start with dot"""
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]


class UploadDataSource(DataSourceProtocol):
    """Upload-based data source for user-uploaded files"""

    def __init__(self, config: UploadDataSourceConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)

    def _get_table_files(self) -> t.Dict[str, Path]:
        """Get mapping of table names to their file paths"""
        tables = {}
        for file_path in self.storage_path.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix in self.config.allowed_extensions
            ):
                # Use filename without extension as table name
                table_name = file_path.stem
                tables[table_name] = file_path
        return tables

    async def list_tables(
        self, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """List all available tables with pagination"""
        try:
            tables = self._get_table_files()
            table_names = sorted(tables.keys())

            total_count = len(table_names)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size

            items = table_names[start_idx:end_idx]

            return PaginatedResponse(
                items=items,
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_next=end_idx < total_count,
                has_previous=page > 1,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to list tables: {e}")

    async def search_tables(
        self, query: str, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """Search for tables matching the query string"""
        try:
            tables = self._get_table_files()
            query_lower = query.lower()

            matching_tables = [
                name for name in tables.keys() if query_lower in name.lower()
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
        except Exception as e:
            raise ConnectionError(f"Failed to search tables: {e}")

    async def to_lazyframe(self, table_name: str) -> pl.LazyFrame:
        """Get a Polars LazyFrame view of the specified table"""
        tables = self._get_table_files()

        if table_name not in tables:
            raise TableNotFoundError(f"Table '{table_name}' not found")

        file_path = tables[table_name]

        try:
            # Determine file type and create appropriate LazyFrame
            if file_path.suffix == ".csv":
                return pl.scan_csv(str(file_path))
            elif file_path.suffix == ".parquet":
                return pl.scan_parquet(str(file_path))
            elif file_path.suffix in [".json", ".jsonl"]:
                # For JSON files, we need to read and then convert to LazyFrame
                # This is less efficient but necessary for JSON
                def _read_json():
                    if file_path.suffix == ".json":
                        return pl.read_json(str(file_path))
                    else:  # .jsonl
                        return pl.read_ndjson(str(file_path))

                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(None, _read_json)
                return df.lazy()
            else:
                raise DataSourceError(f"Unsupported file type: {file_path.suffix}")

        except Exception as e:
            raise DataSourceError(
                f"Failed to create LazyFrame for table '{table_name}': {e}"
            )

    async def get_columns(self, table_name: str) -> pl.Schema:
        """Get the column names and types for a table as a Polars schema"""
        lazy_frame = await self.to_lazyframe(table_name)

        try:
            # Get schema without collecting data
            return lazy_frame.schema

        except Exception as e:
            raise DataSourceError(f"Failed to get schema for table '{table_name}': {e}")

    async def get_table_info(self, table_name: str) -> TableInfo:
        """Get metadata information about a table"""
        tables = self._get_table_files()

        if table_name not in tables:
            raise TableNotFoundError(f"Table '{table_name}' not found")

        file_path = tables[table_name]

        try:
            # Get file stats
            stat = file_path.stat()
            size_bytes = stat.st_size
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Try to get row count
            row_count = None
            try:
                lazy_frame = await self.to_lazyframe(table_name)
                # This will scan the data to count rows - might be expensive
                row_count = lazy_frame.select(pl.len()).collect().item()
            except Exception:
                # If we can't get row count, that's okay
                pass

            return TableInfo(
                name=table_name,
                row_count=row_count,
                size_bytes=size_bytes,
                last_modified=last_modified,
            )

        except Exception as e:
            raise DataSourceError(f"Failed to get info for table '{table_name}': {e}")

    async def upload_file(self, filename: str, content: bytes) -> str:
        """Upload a file and return the table name"""
        file_path = Path(filename)

        # Validate file extension
        if file_path.suffix not in self.config.allowed_extensions:
            raise DataSourceError(f"File extension {file_path.suffix} not allowed")

        # Validate file size
        if len(content) > self.config.max_file_size_mb * 1024 * 1024:
            raise DataSourceError(
                f"File size exceeds {self.config.max_file_size_mb}MB limit"
            )

        # Generate unique filename if file already exists
        target_path = self.storage_path / file_path.name
        counter = 1
        while target_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            target_path = self.storage_path / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            # Try to use aiofiles for async file writing, fallback to sync if not available
            try:
                import aiofiles

                async with aiofiles.open(target_path, "wb") as f:
                    await f.write(content)
            except ImportError:
                # Fallback to synchronous file writing if aiofiles is not available
                def _write_file():
                    with open(target_path, "wb") as f:
                        f.write(content)

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _write_file)

            # Return table name (filename without extension)
            return target_path.stem

        except Exception as e:
            raise DataSourceError(f"Failed to upload file: {e}")


# Convenience factory function
def create_upload_datasource(storage_path: str, **kwargs) -> UploadDataSource:
    """Create an upload data source from storage path and optional configuration"""
    config = UploadDataSourceConfig(storage_path=storage_path, **kwargs)
    return UploadDataSource(config)
