import os
import json
from io import BytesIO
import pandas as pd
from spockflow.inference.config.loader.base import ConfigManager
from spockflow.inference.io.responses import CSVResponse, JSONResponse
from spockflow.inference.io.encoders import (
    content_types,
)

def combine_to_df(result: dict):
    return pd.concat([result['results'], result['paths']], axis=1)
def encode_csv(result: dict):
    result = combine_to_df(result)
    return CSVResponse(content=result.to_csv(header=True, index=False))
def encode_json(result: dict):

    result = combine_to_df(result)
    res = result.to_dict(orient="records")
    if len(res)==1: 
        res = res[0]
    return JSONResponse(res)

encoders = {}
encoders[content_types.JSON] = encode_json
encoders[content_types.CSV] = encode_csv
encoders[content_types.ALL] = encoders[content_types.JSON]



def decode_json(data: bytes):
    return pd.json_normalize(json.loads(data))

def decode_csv(data: bytes):
    return pd.read_csv(BytesIO(data))

decoders = {}
decoders[content_types.JSON] = decode_json
decoders[content_types.CSV] = decode_csv


class ParameterStoreConfigManager(ConfigManager):
    def get_latest_version(self, model_name: str) -> "str":
        return "0.0.0"
    
    def get_config(self, model_name: str, model_version: str):
        with open(os.path.join(os.path.split(__file__)[0], "config.json")) as fp:
            config = json.load(fp)
        

        return {"tree": config}

    def save_to_config(
        self,
        model_name: str,
        model_version: str,
        namespace: str,
        config,
        key,
    ):
        raise NotImplementedError("Cannot yet save to the parameter store")

model_config_cls = ParameterStoreConfigManager
