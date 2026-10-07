import typing as t
from urllib.parse import quote_plus
import asyncio

import polars as pl
from pydantic import Field, field_validator, SecretStr

from .core import (
    DataSourceProtocol,
    DataSourceConfig,
    PaginatedResponse,
    TableInfo,
    DataSourceError,
    TableNotFoundError,
    ConnectionError,
)
from decider import PolarsSchema


class PostgresDataSourceConfig(DataSourceConfig):
    """Configuration for PostgreSQL data source"""

    host: str = Field(..., description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Username")
    password: SecretStr = Field(..., description="Password")
    db_schema: str = Field(
        default="public", description="Database schema", alias="schema"
    )

    # Connection settings
    sslmode: str = Field(default="prefer", description="SSL mode")
    connect_timeout: int = Field(
        default=30, description="Connection timeout in seconds"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number"""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string for Polars"""
        # URL-encode components to handle special characters
        encoded_user = quote_plus(self.username)
        encoded_password = quote_plus(self.password.get_secret_value())
        encoded_host = quote_plus(self.host)
        encoded_database = quote_plus(self.database)

        return (
            f"postgresql://{encoded_user}:{encoded_password}@"
            f"{encoded_host}:{self.port}/{encoded_database}"
            f"?sslmode={self.sslmode}&connect_timeout={self.connect_timeout}"
        )


class PostgresDataSource(DataSourceProtocol):
    """PostgreSQL data source implementation"""

    def __init__(self, config: PostgresDataSourceConfig):
        self.config = config
        self._connection_string = config.get_connection_string()

    async def _run_sync_in_thread(self, func: t.Callable, *args, **kwargs):
        """Helper to run synchronous operations in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _execute_query(self, query: str) -> pl.DataFrame:
        """Execute a SQL query and return results as DataFrame"""
        try:

            def _execute():
                return pl.read_database(query, connection=self._connection_string)

            return await self._run_sync_in_thread(_execute)
        except Exception as e:
            raise ConnectionError(f"Failed to execute query: {e}")

    async def _get_table_names(self) -> t.List[str]:
        """Get list of all table names in the schema"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = $1 AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """

        try:
            # Use direct SQL with schema parameter
            df = await self._execute_query(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{self.config.db_schema}' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            return df["table_name"].to_list()
        except Exception as e:
            raise ConnectionError(f"Failed to get table names: {e}")

    async def list_tables(
        self, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """List all available tables with pagination"""
        table_names = await self._get_table_names()

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

    async def search_tables(
        self, query: str, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """Search for tables matching the query string"""
        table_names = await self._get_table_names()

        # Simple case-insensitive substring search
        query_lower = query.lower()
        matching_tables = [
            table for table in table_names if query_lower in table.lower()
        ]

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

    async def to_lazyframe(self, table_name: str) -> pl.LazyFrame:
        """Get a Polars LazyFrame view of the specified table"""
        # Validate table exists
        table_names = await self._get_table_names()
        if table_name not in table_names:
            raise TableNotFoundError(
                f"Table '{table_name}' not found in schema '{self.config.db_schema}'"
            )

        try:
            # Use scan_database to create a LazyFrame
            # Note: This assumes polars supports PostgreSQL scanning
            query = f'SELECT * FROM "{self.config.db_schema}"."{table_name}"'

            def _create_lazyframe():
                # Polars doesn't have direct scan_database, so we'll create a lazy operation
                # This is a placeholder - in practice you might need to use a different approach
                # or read in chunks for large tables
                return pl.read_database(
                    query, connection=self._connection_string
                ).lazy()

            return await self._run_sync_in_thread(_create_lazyframe)

        except Exception as e:
            raise DataSourceError(
                f"Failed to create LazyFrame for table '{table_name}': {e}"
            )

    async def get_columns(self, table_name: str) -> PolarsSchema:
        """Get the column names and types for a table as a PolarsSchema"""
        # Validate table exists
        table_names = await self._get_table_names()
        if table_name not in table_names:
            raise TableNotFoundError(
                f"Table '{table_name}' not found in schema '{self.config.db_schema}'"
            )

        try:
            # Get column information from information_schema
            query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = '{self.config.db_schema}'
            AND table_name = '{table_name}'
            ORDER BY ordinal_position
            """

            df = await self._execute_query(query)

            # Convert PostgreSQL types to Polars types (simplified mapping)
            type_mapping = {
                "integer": pl.Int32,
                "bigint": pl.Int64,
                "smallint": pl.Int16,
                "numeric": pl.Float64,
                "real": pl.Float32,
                "double precision": pl.Float64,
                "boolean": pl.Boolean,
                "text": pl.Utf8,
                "varchar": pl.Utf8,
                "character": pl.Utf8,
                "date": pl.Date,
                "timestamp": pl.Datetime,
                "timestamptz": pl.Datetime,
            }

            schema_dict = {}
            for row in df.rows():
                column_name, data_type, is_nullable = row
                # Map PostgreSQL type to Polars type
                polars_type = type_mapping.get(data_type, pl.Utf8)  # Default to string
                schema_dict[column_name] = polars_type

            pl_schema = pl.Schema(schema_dict)
            return PolarsSchema.from_polars(pl_schema)

        except Exception as e:
            raise DataSourceError(f"Failed to get schema for table '{table_name}': {e}")

    async def get_table_info(self, table_name: str) -> TableInfo:
        """Get metadata information about a table"""
        # Validate table exists
        table_names = await self._get_table_names()
        if table_name not in table_names:
            raise TableNotFoundError(
                f"Table '{table_name}' not found in schema '{self.config.db_schema}'"
            )

        try:
            # Get table statistics
            stats_query = f"""
            SELECT 
                schemaname, 
                tablename, 
                n_tup_ins + n_tup_upd + n_tup_del as total_rows,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_stat_user_tables 
            WHERE schemaname = '{self.config.db_schema}' AND tablename = '{table_name}'
            """

            try:
                df = await self._execute_query(stats_query)
                if len(df) > 0:
                    row_count = (
                        int(df["total_rows"][0])
                        if df["total_rows"][0] is not None
                        else None
                    )
                    size_bytes = (
                        int(df["size_bytes"][0])
                        if df["size_bytes"][0] is not None
                        else None
                    )
                else:
                    row_count = None
                    size_bytes = None
            except Exception:
                # Fallback: try to get row count with COUNT(*)
                try:
                    count_query = f'SELECT COUNT(*) as count FROM "{self.config.db_schema}"."{table_name}"'
                    count_df = await self._execute_query(count_query)
                    row_count = int(count_df["count"][0])
                    size_bytes = None  # Can't get size without pg_stat_user_tables
                except Exception:
                    row_count = None
                    size_bytes = None

            return TableInfo(
                name=table_name,
                row_count=row_count,
                size_bytes=size_bytes,
                last_modified=None,  # PostgreSQL doesn't track last modified by default
            )

        except Exception as e:
            raise DataSourceError(f"Failed to get info for table '{table_name}': {e}")


# Convenience factory function
def create_postgres_datasource(
    host: str, database: str, username: str, password: str, **kwargs
) -> PostgresDataSource:
    """Create a PostgreSQL data source from connection details"""
    config = PostgresDataSourceConfig(
        host=host,
        database=database,
        username=username,
        password=SecretStr(password),
        **kwargs,
    )
    return PostgresDataSource(config)
