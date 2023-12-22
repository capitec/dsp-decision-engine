#%%
import typing
from pydantic import BaseModel, RootModel, model_validator, PrivateAttr
import re
try:
    from jsonpath_ng.ext.parser import ExtentedJsonPathParser
    from jsonpath_ng import JSONPath, DatumInContext, Index
    from jsonpath_ng.exceptions import JSONPathError, JsonPathParserError
except ImportError:
    # TODO improve below
    raise ImportError("Could not import jsonpath_ng. Please ensure engine is installed with \"pip install engine[\"nested-parser\"]\"")
import pandas as pd


UNIQUE_IDX = re.compile(r"unique_idx\((\d+)\)")

class UniqueIdx(JSONPath):
    """The JSONPath referring to the len of the current object.

    Concrete syntax is '`len`'.
    """

    def __init__(self, method=None):
        m = UNIQUE_IDX.match(method)
        if m is None:
            raise JsonPathParserError("%s is not valid" % method)
        try:
            self.idx = int(m.group(1).strip())
            assert self.idx >= 0
        except (ValueError, AssertionError):
            raise JsonPathParserError("%s must be called with a valid positive integer" % method)

    def find(self, datum):
        datum = DatumInContext.wrap(datum)
        try:
            itr_context = datum
            idx_list = []
            length_list = []
            while itr_context is not None:
                if isinstance(itr_context.path, Index):
                    idx_list.insert(0, itr_context.path.index)
                    length_list.insert(0, len(itr_context.value))
                itr_context = itr_context.context
            if self.idx >= len(length_list):
                raise ValueError("Length of idx greater than the number of lists in the path")
            value = idx_list[0]
            # The [1:] will ensure that the last level does not get multiplied
            for idx_level, (idx,lv) in enumerate(zip(idx_list[1:], length_list),start=1):
                if idx_level > self.idx: break
                value *= lv
                value += idx
        except IndexError:
            return []
        else:
            return [DatumInContext(value,context=None,path=Idx())]

    def __eq__(self, other):
        return isinstance(other, Idx)

    def __str__(self):
        return '`unique_idx`'

    def __repr__(self):
        return f'UniqueIdx({self.idx})'


class Idx(JSONPath):
    """The JSONPath referring to the len of the current object.

    Concrete syntax is '`len`'.
    """

    def find(self, datum):
        datum = DatumInContext.wrap(datum)
        try:
            value = datum.path.index
        except TypeError:
            return []
        else:
            return [DatumInContext(value,context=None,path=Idx())]

    def __eq__(self, other):
        return isinstance(other, Idx)

    def __str__(self):
        return '`idx`'

    def __repr__(self):
        return 'Idx()'

class EngineJsonPathParser(ExtentedJsonPathParser):
    """Custom LALR-parser for JsonPath"""
    def p_jsonpath_named_operator(self, p):
        "jsonpath : NAMED_OPERATOR"
        if p[1] == 'idx':
            p[0] = Idx()
        elif p[1].startswith("unique_idx("):
            p[0] = UniqueIdx(p[1])
        else:
            super(ExtentedJsonPathParser, self).p_jsonpath_named_operator(p)

class JsonPath(RootModel):
    root: str
    _parsed_path: JSONPath = PrivateAttr(None)

    @model_validator(mode='after')
    def valid_json_path(self) -> 'JsonPath':
        try:
            self._parsed_path = EngineJsonPathParser().parse(self.root)
        except JSONPathError as e:
            raise ValueError(f"Invalid JsonPath: {e.args[0] if len(e.args) > 1 else e}") from e
        return self
    
    def find(self, data):
        return self._parsed_path.find(data)
    
    def find_values(self, data):
        return [v.value for v in self.find(data)]


class ParseDictDF(BaseModel):
    fields: typing.Dict[str, JsonPath]
    def get_data_frame(self, payload: typing.Union[dict,list]):
        res = pd.DataFrame({k: v.find_values(payload) for k, v in self.fields.items()})
        # TODO optional typechecking here
        return res
        


class ParseDict(BaseModel):
    data_frames: typing.Dict[str, ParseDictDF]

    def get_data_frames(self, payload: typing.Union[dict,list]):
        return {k: v.get_data_frame(payload) for k, v in self.data_frames.items()}


# %%
