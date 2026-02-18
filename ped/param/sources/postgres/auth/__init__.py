from ._ext import AuthProvider, register_auth_provider
# Preload basic auth providers
from . import basic

__all__ = ['AuthProvider', 'register_auth_provider']