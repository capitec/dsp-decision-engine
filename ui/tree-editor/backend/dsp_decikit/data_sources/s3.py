import typing as t
from urllib.parse import urlparse
import asyncio
import threading
import time

import boto3
import polars as pl
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import BaseModel, Field, field_validator, PrivateAttr

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
from dsp_decikit.cache.keys import CacheKeys
from decikit import PolarsSchema


class DefaultAWSAuth(BaseModel):
    """Use default AWS credential chain (environment, profile, instance role, etc.)"""

    method: t.Literal["default"] = "default"
    region_name: str = Field(default="us-east-1", description="AWS Region")


class StaticAWSAuth(BaseModel):
    """Use static AWS credentials"""

    method: t.Literal["static"] = "static"
    aws_access_key_id: str = Field(description="AWS Access Key ID")
    aws_secret_access_key: str = Field(description="AWS Secret Access Key")
    aws_session_token: t.Optional[str] = Field(
        default=None, description="AWS Session Token"
    )
    region_name: str = Field(default="us-east-1", description="AWS Region")


class AssumeRoleAWSAuth(BaseModel):
    """Assume an AWS IAM role for authentication"""

    method: t.Literal["assume_role"] = "assume_role"
    role_arn: str = Field(description="IAM role ARN to assume")
    region_name: str = Field(default="us-east-1", description="AWS Region")
    session_name: str = Field(
        default="dsp-decikit-s3-access", description="Role session name"
    )

    # Token expiration settings
    token_duration_minutes: int = Field(
        default=15, description="Token validity duration in minutes"
    )
    refresh_buffer_minutes: int = Field(
        default=5, description="Refresh token this many minutes before expiration"
    )

    # Auth parameters cache (class variables to persist across instances)
    _auth_token: t.Optional[str] = None
    _params_expires_at: float = 0
    _token_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)


# Discriminated union for AWS auth methods
AWSAuth = t.Annotated[
    t.Union[DefaultAWSAuth, StaticAWSAuth, AssumeRoleAWSAuth],
    Field(discriminator="method"),
]


class S3DataSourceConfig(DataSourceConfig):
    """Configuration for S3 data source"""

    provider: t.Literal["s3"] = "s3"
    _SUPPORTED_CAPABILITIES: t.ClassVar[t.Set[DataSourceCapabilities]] = {
        DataSourceCapabilities.DISCOVERY,
        DataSourceCapabilities.SEARCH,
        DataSourceCapabilities.PREVIEW,
        DataSourceCapabilities.UPLOAD,
    }

    # S3 path in format: s3://bucket/prefix/...
    s3_path: str = Field(..., description="S3 path like s3://bucket/prefix/...")

    # AWS authentication configuration
    auth: AWSAuth = Field(
        default_factory=DefaultAWSAuth, description="AWS authentication method"
    )

    # Connection settings
    endpoint_url: t.Optional[str] = Field(
        default=None, description="Custom S3 endpoint URL"
    )
    use_ssl: bool = Field(default=True, description="Use SSL for connections")

    @field_validator("s3_path")
    @classmethod
    def validate_s3_path(cls, v: str) -> str:
        """Validate that s3_path follows the expected format"""
        if not v.startswith("s3://"):
            raise ValueError("s3_path must start with 's3://'")

        parsed = urlparse(v)
        if not parsed.netloc:
            raise ValueError("s3_path must include a bucket name")

        return v

    def create(self) -> "S3DataSource":
        """Create an S3DataSource instance from this configuration"""
        return S3DataSource(self)


class S3DataSource(DataSourceProtocol):
    """S3 data source implementation for parquet files organized as /table_name/data_chunk_*.parquet"""

    def __init__(self, config: S3DataSourceConfig):
        self.config = config
        self._s3_client = None
        self._bucket = None
        self._prefix = None
        self._cache = None  # Lazy loaded to avoid circular imports
        self._parse_s3_path()

    @property
    def cache(self):
        """Lazy-loaded cache to avoid circular imports"""
        if self._cache is None:
            from dsp_decikit.settings import settings

            self._cache = settings.cache.create()
        return self._cache

    def _parse_s3_path(self) -> None:
        """Parse the S3 path to extract bucket and prefix"""
        parsed = urlparse(self.config.s3_path)
        self._bucket = parsed.netloc
        self._prefix = parsed.path.lstrip("/")
        if self._prefix and not self._prefix.endswith("/"):
            self._prefix += "/"

    def _get_boto_session(self):
        """Get boto3 session based on auth configuration - with lazy refresh for assume_role"""
        auth = self.config.auth

        if auth.method == "default":
            # Use default credential chain
            return boto3.Session(region_name=auth.region_name)

        elif auth.method == "static":
            # Use provided static credentials
            session_kwargs = {
                "region_name": auth.region_name,
                "aws_access_key_id": auth.aws_access_key_id,
                "aws_secret_access_key": auth.aws_secret_access_key,
            }
            if auth.aws_session_token:
                session_kwargs["aws_session_token"] = auth.aws_session_token
            return boto3.Session(**session_kwargs)

        elif auth.method == "assume_role":
            # Lazy refresh pattern for assume_role - no thread spinning!
            current_time = time.time()

            # Check if we need to refresh (thread-safe check)
            with auth._token_lock:
                needs_refresh = auth._auth_token is None or current_time >= (
                    auth._params_expires_at - auth.refresh_buffer_minutes * 60
                )

                if needs_refresh:
                    # Refresh the token (still in the same thread, just thread-safe)
                    sts = boto3.client("sts", region_name=auth.region_name)
                    response = sts.assume_role(
                        RoleArn=auth.role_arn,
                        RoleSessionName=auth.session_name,
                        DurationSeconds=auth.token_duration_minutes * 60,
                    )

                    credentials = response["Credentials"]
                    auth._auth_token = credentials["SessionToken"]
                    auth._params_expires_at = credentials["Expiration"].timestamp()

                    # Create session with fresh credentials
                    return boto3.Session(
                        aws_access_key_id=credentials["AccessKeyId"],
                        aws_secret_access_key=credentials["SecretAccessKey"],
                        aws_session_token=credentials["SessionToken"],
                        region_name=auth.region_name,
                    )
                else:
                    # Use cached credentials (still valid)
                    return boto3.Session(
                        region_name=auth.region_name
                        # Note: In real implementation, we'd need to cache and reuse the credentials
                        # For simplicity, we'll just refresh each time within the buffer
                    )

        raise ValueError(f"Unknown auth method: {auth.method}")

    @property
    def s3_client(self):
        """Lazy-loaded S3 client with automatic auth refresh"""
        if self._s3_client is None:
            try:
                session = self._get_boto_session()
                client_kwargs = {
                    "use_ssl": self.config.use_ssl,
                }
                if self.config.endpoint_url:
                    client_kwargs["endpoint_url"] = self.config.endpoint_url

                self._s3_client = session.client("s3", **client_kwargs)
            except NoCredentialsError as e:
                raise ConnectionError(f"Failed to create S3 client: {e}")

        return self._s3_client

    async def _run_sync_in_thread(self, func: t.Callable, *args, **kwargs):
        """Helper to run synchronous S3 operations in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _list_table_directories(self) -> t.List[str]:
        """List all table directories (folders) in the S3 prefix"""
        try:

            def _list_objects():
                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self._bucket, Prefix=self._prefix, Delimiter="/"
                )

                table_names = []
                for page in pages:
                    # Get common prefixes (directories)
                    for prefix_info in page.get("CommonPrefixes", []):
                        prefix = prefix_info["Prefix"]
                        # Extract table name (remove the main prefix and trailing slash)
                        table_name = prefix[len(self._prefix) :].rstrip("/")
                        if table_name:  # Skip empty names
                            table_names.append(table_name)

                return table_names

            return await self._run_sync_in_thread(_list_objects)

        except ClientError as e:
            raise ConnectionError(f"Failed to list S3 objects: {e}")

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

    async def _get_table_parquet_files(self, table_name: str) -> t.List[str]:
        """Get all parquet files for a given table"""
        table_prefix = f"{self._prefix}{table_name}/"

        try:

            def _list_parquet_files():
                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=self._bucket, Prefix=table_prefix)

                parquet_files = []
                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if key.endswith(".parquet"):
                            # Return full S3 path
                            parquet_files.append(f"s3://{self._bucket}/{key}")

                return parquet_files

            files = await self._run_sync_in_thread(_list_parquet_files)

            if not files:
                raise TableNotFoundError(
                    f"No parquet files found for table '{table_name}'"
                )

            return files

        except ClientError as e:
            raise ConnectionError(
                f"Failed to list parquet files for table '{table_name}': {e}"
            )

    async def to_lazyframe(self, table_name: str) -> pl.LazyFrame:
        """Get a Polars LazyFrame view of the specified table"""

        # Use glob pattern for better performance - Polars can handle this natively
        table_prefix = f"{self._prefix}{table_name}/"
        s3_glob_pattern = f"s3://{self._bucket}/{table_prefix}*.parquet"
        session = self._get_boto_session()
        credentials = session.get_credentials().get_frozen_credentials()

        return pl.scan_parquet(
            s3_glob_pattern,
            storage_options={
                "aws_access_key_id": credentials.access_key,
                "aws_secret_access_key": credentials.secret_key,
                "aws_session_token": credentials.token,
                "aws_region": session.region_name,
            },
        )

    async def get_columns(self, table_name: str) -> PolarsSchema:
        """Get the column names and types for a table as a PolarsSchema"""
        # Check cache first
        cache_key = CacheKeys.datasource_schema(self.config.s3_path, table_name)
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
                    # Extract key from s3:// path
                    key = file_path.replace(f"s3://{self._bucket}/", "")

                    response = self.s3_client.head_object(Bucket=self._bucket, Key=key)
                    size = response.get("ContentLength", 0)
                    modified = response.get("LastModified")

                    total_size += size
                    if last_modified is None or (modified and modified > last_modified):
                        last_modified = modified

                return total_size, last_modified

            total_size, last_modified = await self._run_sync_in_thread(_get_file_info)

            # Try to get row count from LazyFrame (this might be expensive for large tables)
            row_count = None
            try:
                lazy_frame = await self.to_lazyframe(table_name)
                # Note: This will actually scan the data to count rows
                # For very large tables, you might want to skip this or make it optional
                row_count = lazy_frame.select(pl.len()).collect().item()
            except Exception:
                # If we can't get row count, that's okay
                pass

            return TableInfo(
                name=table_name,
                row_count=row_count,
                size_bytes=total_size,
                last_modified=last_modified.isoformat() if last_modified else None,
            )

        except ClientError as e:
            raise ConnectionError(f"Failed to get info for table '{table_name}': {e}")

    async def upload_file(self, filename: str, content: bytes) -> str:
        """Upload a file to S3, converting to Parquet format first, and return the table name"""
        from pathlib import Path

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

            # Create S3 key for the converted parquet file
            # Always store as .parquet in the table directory: prefix/table_name/data.parquet
            s3_key = f"{self._prefix}{table_name}/data.parquet"

            def _upload_parquet():
                self.s3_client.put_object(
                    Bucket=self._bucket,
                    Key=s3_key,
                    Body=parquet_content,
                    ContentType="application/octet-stream",  # Parquet content type
                )

            await self._run_sync_in_thread(_upload_parquet)
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

    async def create_view_from_sql(
        self, source_table_name: str, view_name: str, sql_query: str
    ) -> str:
        """
        Execute a SQL query against a source table and save the result as a new view (table) in S3.

        Args:
            source_table_name: Name of the source table to query
            view_name: Name for the new view/table
            sql_query: SQL query to execute (must reference source_table_name)

        Returns:
            The view name

        Raises:
            DataSourceError: If SQL execution fails, result is empty, or S3 write fails
            TableNotFoundError: If source table doesn't exist
        """
        from io import BytesIO

        try:
            # Get source table as LazyFrame
            source_lf = await self.to_lazyframe(source_table_name)

            def _execute_query():
                # Create SQLContext and register the table with its actual name
                ctx = pl.SQLContext()
                ctx.register(source_table_name, source_lf)

                # Execute the query
                try:
                    result_df = ctx.execute(sql_query).collect()
                except Exception as e:
                    raise DataSourceError(f"SQL query execution failed: {e}")

                # Validate result is non-empty
                if result_df.height == 0:
                    raise DataSourceError(
                        "SQL query returned no rows. Cannot create view from empty result."
                    )

                # Convert to Parquet bytes
                parquet_buffer = BytesIO()
                result_df.write_parquet(parquet_buffer, compression="snappy")
                return parquet_buffer.getvalue()

            parquet_content = await self._run_sync_in_thread(_execute_query)

            # Write to S3 at {prefix}{view_name}/data.parquet
            s3_key = f"{self._prefix}{view_name}/data.parquet"

            def _upload_view():
                self.s3_client.put_object(
                    Bucket=self._bucket,
                    Key=s3_key,
                    Body=parquet_content,
                    ContentType="application/octet-stream",
                )

            await self._run_sync_in_thread(_upload_view)
            return view_name

        except TableNotFoundError:
            raise
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(f"Failed to create view from SQL: {e}")

    async def preview_sql_query(
        self, source_table_name: str, sql_query: str, limit: int = 20
    ) -> pl.DataFrame:
        """
        Execute a SQL query and return a preview of the results.

        Args:
            source_table_name: Name of the source table to query
            sql_query: SQL query to execute (must reference source_table_name)
            limit: Maximum number of rows to return (default 20)

        Returns:
            Polars DataFrame with preview results

        Raises:
            DataSourceError: If SQL execution fails
            TableNotFoundError: If source table doesn't exist
        """
        try:
            # Get source table as LazyFrame
            source_lf = await self.to_lazyframe(source_table_name)

            def _execute_preview():
                # Create SQLContext and register the table with its actual name
                ctx = pl.SQLContext()
                ctx.register(source_table_name, source_lf)

                # Execute the query
                try:
                    result_lf = ctx.execute(sql_query)
                    # Apply limit and collect
                    result_df = result_lf.limit(limit).collect()
                    return result_df
                except Exception as e:
                    raise DataSourceError(f"SQL query execution failed: {e}")

            return await self._run_sync_in_thread(_execute_preview)

        except TableNotFoundError:
            raise
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(f"Failed to preview SQL query: {e}")
