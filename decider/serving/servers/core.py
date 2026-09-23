import json
import typing as t

from decider.exceptions import DeciderError

_INITIALIZING = b'{"message": "Server is initializing, please try again shortly."}'


def error_response(error: DeciderError) -> t.Tuple[int, bytes, str]:
    body = error.get_response_body()
    payload: t.Dict[str, str] = {"message": body.message}
    if body.details:
        payload["details"] = body.details
    return error.get_status_code(), json.dumps(payload).encode(), "application/json"


def parse_content_headers(headers: t.Mapping[str, str]) -> t.Tuple[str, str]:
    """Return (content_type, accept) from a headers mapping."""
    return headers.get("content-type", ""), headers.get("accept", "*/*")
