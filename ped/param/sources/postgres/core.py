import typing as t
import os
from pydantic import BaseModel, Field
from .auth import AuthProvider


class DatabaseSettings(BaseModel):
    """Database connection and configuration settings"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    auth: AuthProvider = Field(default_factory=lambda: {"method": "basic"})
    
    # Table configuration
    version_table: t.Optional[str] = Field(default=None, description="Optional version table name")
    parameter_table: str = Field(description="Parameter table name")
    
    # Environment configuration (used when version_table is set)
    environment: str = Field(default_factory=lambda: os.environ.get('ENV', 'dev'))
    
    def get_connection_params(self) -> t.Dict[str, t.Any]:
        """Get connection parameters for psycopg3"""
        return self.auth.get_connection_params(self)