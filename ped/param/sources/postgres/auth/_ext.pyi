import typing as t
from .core import BaseAuth
from ped._ext import TExtendableModel

AuthProvider = TExtendableModel[BaseAuth]

def register_auth_provider(provider_class: t.Type[BaseAuth]) -> None: ...

__all__: list[str]