import typing as t
from io import BytesIO
import polars as pl
from .media_types import MediaType

class Response(t.NamedTuple):
    content: bytes
    media_type: t.Optional[str] = None



def format_application_json(result: pl.DataFrame) -> "Response":
    if len(result) == 1:
        return Response(
            content=result.write_json().removeprefix('[').removesuffix(']').encode("utf-8"),
            media_type=MediaType.APPLICATION_JSON.value
        )
    else:
        return Response(content=result.write_json().encode("utf-8"), media_type=MediaType.APPLICATION_JSON.value)

def format_application_jsonl(result: pl.DataFrame) -> "Response":
    return Response(
        content=result.write_ndjson().encode("utf-8"),
        media_type=MediaType.APPLICATION_JSONL.value
    )

def format_application_x_parquet(result: pl.DataFrame) -> "Response":
    f = BytesIO()
    result.write_parquet(f)
    return Response(
        content=f.getvalue(),
        media_type=MediaType.APPLICATION_X_PARQUET.value
    )

def format_text_csv(result: pl.DataFrame) -> "Response":
    return Response(
        content=result.write_csv().encode("utf-8"),
        media_type=MediaType.TEXT_CSV.value
    )

DEFAULT_OUTPUT_FORMATTERS = {
    MediaType.ANY.value: format_application_json,  # Default to JSON for any Accept header
    MediaType.APPLICATION_JSON.value: format_application_json,
    MediaType.APPLICATION_JSONL.value: format_application_jsonl,
    MediaType.APPLICATION_X_PARQUET.value: format_application_x_parquet,
    MediaType.TEXT_CSV.value: format_text_csv,
}