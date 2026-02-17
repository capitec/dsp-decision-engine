import typing as t
from abc import ABC, abstractmethod
from pydantic import BaseModel

if t.TYPE_CHECKING:
    from ..core import DatabaseSettings


class BaseAuth(BaseModel, ABC):
    """Base class for all authentication providers"""
    method: str
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'method' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'method' field")

    @abstractmethod
    def get_connection_params(self, db_settings: "DatabaseSettings") -> t.Dict[str, t.Any]:
        """Get connection parameters for psycopg3"""
        ...

    def invalidate_auth(self):
        """Invalidate any cached auth tokens/credentials"""
        pass