import json
import pandas as pd
from io import BytesIO


# TODO Will need to think a bit about what is desired the most
# pd.json_normalize(json.loads(data)).to_dict("series")?
# Maybe a middle ground like automatically converting all list to pd.Series with a custom decode hook
def decode_json(data: bytes):
    return json.loads(data)


def decode_csv(data: bytes):
    # TODO performance between this and a csv reader
    return pd.read_csv(BytesIO(data)).to_dict(orient="list")


default_decoders = {}
default_decoders["application/json"] = decode_json
default_decoders["text/csv"] = decode_csv
