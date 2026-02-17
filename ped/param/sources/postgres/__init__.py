from .source import PostgresSource
from .core import DatabaseSettings
from .auth import AuthProvider, register_auth_provider

# Register the source with the main sources system
from .._ext import register_source
register_source(PostgresSource)

__all__ = ['PostgresSource', 'DatabaseSettings', 'AuthProvider', 'register_auth_provider']