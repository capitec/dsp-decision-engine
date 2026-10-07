from .routes import router
from .server import ServerInfoResponse

TAGS = [
    {"name": "server", "description": "Server information operations"},
]

__all__ = ["router", "TAGS", "ServerInfoResponse"]
