import typing as t
from pydantic import SecretStr
from .core import BaseAuth
from ._ext import register_auth_provider

if t.TYPE_CHECKING:
    from ..core import DatabaseSettings


class BasicAuth(BaseAuth):
    """Basic username/password authentication"""
    method: t.Literal["basic"] = "basic"
    username: str = "postgres"
    password: SecretStr = SecretStr("postgres")

    def get_connection_params(self, db_settings: "DatabaseSettings") -> t.Dict[str, t.Any]:
        """Get connection parameters for psycopg3"""
        return {
            "host": db_settings.host,
            "port": db_settings.port,
            "dbname": db_settings.database,
            "user": self.username,
            "password": self.password.get_secret_value(),
        }


register_auth_provider(BasicAuth)