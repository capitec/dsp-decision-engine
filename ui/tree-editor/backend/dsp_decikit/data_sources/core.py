import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from decikit import PolarsSchema


@dataclass
class TableInfo:
    """Information about a table in the datasource"""

    name: str
    row_count: t.Optional[int] = None
    size_bytes: t.Optional[int] = None
    last_modified: t.Optional[str] = None


@dataclass
class PaginatedResponse:
    """Paginated response for table listings"""

    items: t.List[str]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class DataSourceInfo(BaseModel):
    """Information about a configured data source"""

    name: str
    provider: str
    display_name: str
    description: t.Optional[str] = None
    icon: t.Optional[str] = None
    capabilities: t.Set["DataSourceCapabilities"]


class DataSourceList(BaseModel):
    """List of configured data sources"""

    datasources: t.List[DataSourceInfo]


class DataSourceTablesResponse(BaseModel):
    """Response containing tables from a data source"""

    datasource: str
    tables: t.List[str]
    pagination: "PaginationInfo"


class DataSourceSearchResponse(BaseModel):
    """Response containing search results for tables from a data source"""

    datasource: str
    query: str
    tables: t.List[str]
    pagination: "PaginationInfo"


class PaginationInfo(BaseModel):
    """Pagination information"""

    page: int
    page_size: int
    total_count: int
    has_next: bool
    has_previous: bool


class TableDetailResponse(BaseModel):
    """Detailed information about a table"""

    datasource: str
    table: str
    info: TableInfo
    table_schema: PolarsSchema


class TablePreviewData(BaseModel):
    """Preview data for a table"""

    columns: t.List[str]
    rows: t.List[t.List[t.Any]]
    row_count: int
    total_columns: int


class TablePreviewDataColumnar(BaseModel):
    """Columnar preview data for more efficient transfer of large datasets"""

    columns: t.List[str]
    data: t.Dict[str, t.List[t.Any]]  # Column name -> column values
    row_count: int
    total_columns: int


class TablePreviewResponseColumnar(BaseModel):
    """Response containing table preview data in columnar format"""

    datasource: str
    table: str
    limit: int
    data: TablePreviewDataColumnar


class TablePreviewResponse(BaseModel):
    """Response containing table preview data"""

    datasource: str
    table: str
    limit: int
    data: TablePreviewData


class DataSourceProtocol(ABC):
    """Protocol for data sources that can provide table listings and data access"""

    @abstractmethod
    async def list_tables(
        self, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """
        List all available tables in the datasource with pagination

        Args:
            page: Page number (1-based)
            page_size: Number of tables per page

        Returns:
            PaginatedResponse with table names
        """
        pass

    @abstractmethod
    async def search_tables(
        self, query: str, page: int = 1, page_size: int = 100
    ) -> PaginatedResponse:
        """
        Search for tables matching the query string

        Args:
            query: Search query string
            page: Page number (1-based)
            page_size: Number of tables per page

        Returns:
            PaginatedResponse with matching table names
        """
        pass

    @abstractmethod
    async def to_lazyframe(self, table_name: str) -> pl.LazyFrame:
        """
        Get a Polars LazyFrame view of the specified table

        Args:
            table_name: Name of the table to load

        Returns:
            Polars LazyFrame for the table
        """
        pass

    @abstractmethod
    async def get_columns(self, table_name: str) -> PolarsSchema:
        """
        Get the column names and types for a table as a PolarsSchema

        Args:
            table_name: Name of the table

        Returns:
            PolarsSchema with column information
        """
        pass

    @abstractmethod
    async def get_table_info(self, table_name: str) -> TableInfo:
        """
        Get metadata information about a table

        Args:
            table_name: Name of the table

        Returns:
            TableInfo with metadata
        """
        pass


class DataSourceCapabilities(str, Enum):
    """Capabilities that a data source can support"""

    UPLOAD = "upload"
    DISCOVERY = "discovery"
    SEARCH = "search"
    PREVIEW = "preview"


class DataSourceConfig(BaseModel):
    """Base configuration class for data sources"""

    model_config = ConfigDict(extra="forbid")
    _SUPPORTED_CAPABILITIES: t.ClassVar[t.Set[DataSourceCapabilities]] = set()
    display_name: t.Optional[str] = None
    description: t.Optional[str] = None
    icon: t.Optional[str] = None
    disable_capabilities: t.Set[DataSourceCapabilities] = Field(
        default_factory=set,
        description="Set of capabilities to disable for this data source",
    )
    max_preview_rows: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of rows to return in data previews",
    )
    cache_ttl_seconds: int = Field(
        default=600,
        ge=0,
        description="Cache TTL for preview data in seconds (0 = no caching)",
    )
    schema_cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL for schema data in seconds (0 = no caching)",
    )

    def capabilities(self) -> t.Set[DataSourceCapabilities]:
        """Get the set of capabilities this data source supports"""
        return self._SUPPORTED_CAPABILITIES - self.disable_capabilities


class DataSourceError(Exception):
    """Base exception for data source operations"""

    pass


class TableNotFoundError(DataSourceError):
    """Raised when a requested table is not found"""

    pass


class DataSourceNotFoundError(DataSourceError):
    """Raised when a requested datasource is not found or configured"""

    pass


class ConnectionError(DataSourceError):
    """Raised when connection to data source fails"""

    pass
