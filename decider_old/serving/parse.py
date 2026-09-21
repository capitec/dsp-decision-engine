import json
import typing as t
import polars as pl
from dataclasses import dataclass, field
from .media_types import MediaType


@dataclass(slots=True)
class ParserConfig:
    input_frame_keys: t.List[str] = field(default_factory=lambda: ["input"])


TParsedMessage = t.Dict[str, pl.DataFrame]


async def parse_application_json(data: bytes, parse_config: ParserConfig) -> TParsedMessage:
    if len(parse_config.input_frame_keys) == 1:
        return {"input": pl.read_json(data)}
    else:
        json_dict = json.loads(data)
        return {key: pl.from_dict(json_dict[key]) for key in parse_config.input_frame_keys}


async def parse_application_jsonl(data: bytes, parse_config: ParserConfig) -> TParsedMessage:
    assert len(parse_config.input_frame_keys) == 1, "JSONL parsing only supports a single input frame key"
    return {parse_config.input_frame_keys[0]: pl.read_ndjson(data)}

async def parse_application_x_parquet(data: bytes, parse_config: ParserConfig) -> TParsedMessage:
    assert len(parse_config.input_frame_keys) == 1, "Parquet parsing only supports a single input frame key"
    return {parse_config.input_frame_keys[0]: pl.read_parquet(data)}

async def parse_text_csv(data: bytes, parse_config: ParserConfig) -> TParsedMessage:
    assert len(parse_config.input_frame_keys) == 1, "CSV parsing only supports a single input frame key"
    return {parse_config.input_frame_keys[0]: pl.read_csv(data)}

async def parse_application_excel(data: bytes, parse_config: ParserConfig) -> TParsedMessage:
    assert len(parse_config.input_frame_keys) == 1, "Excel parsing only supports a single input frame key"
    return {parse_config.input_frame_keys[0]: pl.read_excel(data)}



DEFAULT_INPUT_HANDLERS = {
    MediaType.APPLICATION_JSON.value: parse_application_json,
    MediaType.APPLICATION_JSONL.value: parse_application_jsonl,
    MediaType.APPLICATION_X_PARQUET.value: parse_application_x_parquet,
    MediaType.TEXT_CSV.value: parse_text_csv,
    MediaType.APPLICATION_EXCEL.value: parse_application_excel,
    MediaType.APPLICATION_VND_MS_EXCEL.value: parse_application_excel,
}
