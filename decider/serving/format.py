import typing as t
from io import BytesIO

import polars as pl


class Response(t.NamedTuple):
    """What `RequestHandler.output_fn` returns: the body and its media type.

    Example::

        Response(b'{"capped": 48.0}', "application/json")
    """

    content: bytes
    media_type: str


def _json(result: pl.DataFrame) -> bytes:
    text = result.write_json()
    # One row goes back as an object, not a one-element list.
    return (text.removeprefix("[").removesuffix("]") if len(result) == 1 else text).encode()


def _parquet(result: pl.DataFrame) -> bytes:
    f = BytesIO()
    result.write_parquet(f)
    return f.getvalue()


# Accept header -> (response media type, writer).
DEFAULT_OUTPUT_FORMATTERS: dict[str, tuple[str, t.Callable[[pl.DataFrame], bytes]]] = {
    "*/*": ("application/json", _json),
    "application/json": ("application/json", _json),
    "application/jsonl": ("application/jsonl", lambda r: r.write_ndjson().encode()),
    "application/x-parquet": ("application/x-parquet", _parquet),
    "text/csv": ("text/csv", lambda r: r.write_csv().encode()),
}
