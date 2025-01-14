import pandas as pd
from typing import Any

from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue


values_schema = core_schema.dict_schema(keys_schema=core_schema.str_schema())

dataframe_json_schema = core_schema.model_fields_schema(
    {
        "type": core_schema.model_field(core_schema.literal_schema(["DataFrame"])),
        "values": core_schema.model_field(
            core_schema.union_schema(
                [values_schema, core_schema.list_schema(values_schema)]
            )
        ),
        "dtypes": core_schema.model_field(
            core_schema.with_default_schema(
                core_schema.union_schema(
                    [
                        core_schema.none_schema(),
                        core_schema.dict_schema(
                            keys_schema=core_schema.str_schema(),
                            values_schema=core_schema.str_schema(),
                        ),
                    ]
                ),
                default=None,
            )
        ),
    }
)
series_json_schema = core_schema.model_fields_schema(
    {
        "type": core_schema.model_field(core_schema.literal_schema(["Series"])),
        "values": core_schema.model_field(core_schema.list_schema()),
        "name": core_schema.model_field(
            core_schema.with_default_schema(
                core_schema.union_schema(
                    [core_schema.none_schema(), core_schema.str_schema()]
                ),
                default=None,
            )
        ),
    }
)


def validate_to_dataframe_schema(value: dict) -> pd.DataFrame:
    value, *_ = value
    values = value["values"]
    if isinstance(values, dict):
        values = [values]
    df = pd.DataFrame(values)
    dtypes = value.get("dtypes")
    if dtypes:
        return df.astype(dtypes)
    return df


def validate_to_series_schema(value: dict) -> pd.Series:
    value, *_ = value
    return pd.Series(value["values"], name=value.get("name"))


from_df_dict_schema = core_schema.chain_schema(
    [
        dataframe_json_schema,
        # core_schema.dict_schema(),
        core_schema.no_info_plain_validator_function(validate_to_dataframe_schema),
    ]
)
from_series_dict_schema = core_schema.chain_schema(
    [
        series_json_schema,
        # core_schema.dict_schema(),
        core_schema.no_info_plain_validator_function(validate_to_series_schema),
    ]
)


def dump_df_to_dict(instance: pd.DataFrame) -> dict:
    values = instance.to_dict(orient="records")
    if len(values) == 1:
        values = values[0]
    return {
        "type": "DataFrame",
        "values": values,
        "dtypes": {k: str(v) for k, v in instance.dtypes.items()},
    }


def dump_series_to_dict(instance: pd.Series) -> dict:
    return {"type": "Series", "values": instance.to_list(), "name": instance.name}


class _PandasDataFramePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:

        return core_schema.json_or_python_schema(
            json_schema=from_df_dict_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(pd.DataFrame),
                    from_df_dict_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                dump_df_to_dict
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(dataframe_json_schema)


class _PandasSeriesPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:

        return core_schema.json_or_python_schema(
            json_schema=from_series_dict_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(pd.Series),
                    from_series_dict_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                dump_series_to_dict
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(dataframe_json_schema)


DataFrame = Annotated[pd.DataFrame, _PandasDataFramePydanticAnnotation]
Series = Annotated[pd.Series, _PandasSeriesPydanticAnnotation]
