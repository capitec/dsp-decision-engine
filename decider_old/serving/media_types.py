from enum import Enum

class MediaType(Enum):
    """A convenience enum to represent common url media types."""
    ANY = "*/*"
    APPLICATION_JSON = "application/json"
    APPLICATION_JSONL = "application/jsonl"
    APPLICATION_X_PARQUET = "application/x-parquet"
    TEXT_CSV = "text/csv"
    APPLICATION_EXCEL = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    APPLICATION_VND_MS_EXCEL = "application/vnd.ms-excel"
