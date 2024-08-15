import pandas as pd
from typing import Any, Union

from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue


def dump_to_dict(instance: Union[pd.Series, pd.DataFrame]) -> dict:
    if isinstance(instance, pd.DataFrame):
        return {"type": "DataFrame", "values": instance.to_dict(orient="list")}
    return {"values": instance.to_list(), "name": instance.name, "type": "Series"}


# class Series(pd.Series):
#     @classmethod
#     def __get_pydantic_core_schema__(
#         cls, __source: type[Any], __handler: GetCoreSchemaHandler
#     ) -> core_schema.CoreSchema:
#         return core_schema.no_info_before_validator_function(
#             pd.Series,
#             core_schema.dict_schema(),
#             serialization=core_schema.plain_serializer_function_ser_schema(dump_to_dict)
#         )

# class DataFrame(pd.DataFrame):
#     @classmethod
#     def __get_pydantic_core_schema__(
#         cls, __source: type[Any], __handler: GetCoreSchemaHandler
#     ) -> core_schema.CoreSchema:
#         return core_schema.no_info_before_validator_function(
#             pd.DataFrame,
#             core_schema.dict_schema(),
#             serialization=core_schema.plain_serializer_function_ser_schema(dump_to_dict)
#         )

# This class allows Pydantic to serialise and deserialise pandas Dataframes and Series items


class _PandasPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_from_dict(value: dict) -> pd.Series:
            if value.get("type") == "DataFrame":
                return pd.DataFrame(value["values"])
            else:
                return pd.Series(value["values"], name=value["name"])

        from_int_schema = core_schema.chain_schema(
            [
                core_schema.dict_schema(),  # TODO make this more comprehensive
                core_schema.no_info_plain_validator_function(validate_from_dict),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_int_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(pd.Series),
                    from_int_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                dump_to_dict
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.dict_schema())  # TODO make this more comprehensive


DataFrame = Annotated[pd.DataFrame, _PandasPydanticAnnotation]


Series = Annotated[pd.Series, _PandasPydanticAnnotation]
