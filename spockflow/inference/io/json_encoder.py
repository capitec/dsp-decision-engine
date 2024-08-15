import numpy as np
import pandas as pd
from json import JSONEncoder


class PandasJsonEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, pd.Series):
            return o.to_list()
        elif isinstance(o, pd.DataFrame):
            return {k: v.to_list() for k, v in o.items()}
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.tolist()
        return o.__dict__
