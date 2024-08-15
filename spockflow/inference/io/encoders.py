import typing
from json import JSONEncoder

import numpy as np
import pandas as pd

from .responses import Response, CSVResponse, JSONResponse
from . import content_types


TDefaultResult = typing.Union[
    pd.Series, pd.DataFrame, np.ndarray, typing.Dict[str, typing.Any]
]


class JsonEncoder:
    def __init__(self, encoder=None):
        self._encoder = encoder

    @property
    def encoder(self) -> "JSONEncoder":
        if self._encoder is None:
            from .json_encoder import PandasJsonEncoder

            self._encoder = PandasJsonEncoder()
        return self._encoder

    def __call__(self, data: "TDefaultResult") -> "Response":
        return Response(self.encoder.encode(data), media_type="application/json")


def encode_csv(result: "TDefaultResult") -> "Response":
    from io import BytesIO

    stream = BytesIO()
    if isinstance(result, dict):
        from hamilton.base import PandasDataFrameResult

        result = PandasDataFrameResult.build_result()
    if callable(getattr(result, "to_csv", None)):
        result.to_csv(stream)
    else:
        np.savetxt(stream, result, delimiter=",", fmt="%s")
    stream.seek(0)
    return CSVResponse(stream.read())


default_encoders = {}
default_encoders[content_types.JSON] = JsonEncoder()
default_encoders[content_types.CSV] = encode_csv
default_encoders[content_types.ALL] = default_encoders[content_types.JSON]
