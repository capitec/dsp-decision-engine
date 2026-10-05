import json
import typing as t

from decider.exceptions import DeciderError

_INITIALIZING = b'{"message": "Server is initializing, please try again shortly."}'


def ready(handler: t.Any) -> bool:
    return handler is not None and handler.active is not None


def error_response(error: DeciderError) -> t.Tuple[int, bytes, str]:
    return error._STATUS_CODE, json.dumps({"message": str(error)}).encode(), "application/json"


def parse_content_headers(headers: t.Mapping[str, str]) -> t.Tuple[str, str]:
    """Return (content_type, accept) from a headers mapping."""
    return headers.get("content-type", ""), headers.get("accept", "*/*")
