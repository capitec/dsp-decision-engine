"""
This module enables auth providers to be extended by external packages without creating hard dependencies.
It follows the same extensible pattern used for sources and other components.
"""
from .core import BaseAuth
from ped._ext import create_extendable_model

# Create the extensible auth model
AuthProvider, register_auth_provider = create_extendable_model(
    BaseAuth,
    discriminator_field="method", 
    model_name="AuthProvider"
)

__all__ = ['BaseAuth', 'AuthProvider', 'register_auth_provider']