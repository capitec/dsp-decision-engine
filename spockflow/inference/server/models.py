import typing

from pydantic import BaseModel, model_validator
from spockflow.inference.io import content_types
from spockflow.inference.config.loader.base import TNamespacedConfig


class PredictWithConfigRequestModel(BaseModel):
    payload: typing.Union[str, bytes]
    config: TNamespacedConfig
    content_type: str = content_types.JSON

    @model_validator(mode="before")
    def check_either_payload_or_b64payload(cls, data):
        assert (
            "b64payload" not in data or "payload" not in data
        ), "both 'b64payload' and 'payload' should not be included."
        if "b64payload" in data:
            from base64 import b64decode

            data["payload"] = b64decode(data.pop("b64payload"))
        return data
