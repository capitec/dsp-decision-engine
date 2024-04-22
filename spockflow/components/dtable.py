import typing
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field
from dataclasses import dataclass

from .._util import get_name, safe_update

DType = typing.TypeVar("DType", bound=np.generic)

ArrayP = typing.Annotated[npt.NDArray[DType], typing.Literal["P"]]
ArrayV = typing.Annotated[npt.NDArray[DType], typing.Literal["V"]]
ArrayPxV = typing.Annotated[npt.NDArray[DType], typing.Literal["P", "V"]]

# registered_dtable_ops = []

class DecisionTableOp(BaseModel):
    # _sub_classes: typing.ClassVar[set] = set()
    predicate: typing.List[typing.Union[int, float, str]]
    # def __init_subclass__(cls, /, **kwargs: object) -> None:
    #     # super().__init_subclass__(**kwargs)
    #     registered_dtable_ops.extend(cls)

    @property
    def pred(self):
        return np.array(self.predicate)
    
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        raise NotImplementedError()

class DTMin(DecisionTableOp):
    op: typing.Literal["MIN"] = "MIN"
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] >= self.pred[:,None]
        criteria_mask[np.isnan(self.pred)]=True
        return criteria_mask
    def add_trace(self, idx):
        return {"lower": self.pred[idx]}

class DTMax(DecisionTableOp):
    op: typing.Literal["MAX"] = "MAX"
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] < self.pred[:,None]
        criteria_mask[np.isnan(self.pred)]=True # TODO verify that this works for isnull
        return criteria_mask
    def add_trace(self, idx):
        return {"upper": self.pred[idx]}
    
class DTMinNotEq(DecisionTableOp):
    """DTMIN is the default of min inclusive this is defined to be not inclusive of the lower bound."""
    op: typing.Literal["MINNEQ"] = "MINNEQ"
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] > self.pred[:,None]
        criteria_mask[np.isnan(self.pred)]=True
        return criteria_mask
    def add_trace(self, idx):
        return {"lower": self.pred[idx]}

class DTMaxEq(DecisionTableOp):
    op: typing.Literal["MAXEQ"] = "MAXEQ"
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] <= self.pred[:,None]
        criteria_mask[np.isnan(self.pred)]=True # TODO verify that this works for isnull
        return criteria_mask
    def add_trace(self, idx):
        return {"upper": self.pred[idx]}

class DTIn(DecisionTableOp):
    op: typing.Literal["IN"] = "IN"
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        pred = self.pred
        return np.apply_along_axis( # TODO maybe write this as a numba function
            lambda v: np.isin(values,eval(v[0])) if not pd.isnull(v[0]) else [True]*len(values), 
            0, pred[None]
        )
    def add_trace(self, idx):
        return {"in": self.pred[idx]}


class DTEq(DecisionTableOp):
    op: typing.Literal["EQ"] = "EQ"
    tol: typing.Optional[float] = 1e-9
    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        if self.tol is not None: # Set as a default
            criteria_mask = np.abs(values[None] - self.pred[:,None]) < self.tol
        else:
            criteria_mask = values[None] == self.pred[:,None]
        criteria_mask[np.isnan(self.pred)]=True
        return criteria_mask
    def add_trace(self, idx):
        return {"equals": self.pred[idx]}

RegDTableOps = typing.Annotated[
    typing.Union[DTMin,DTMax,DTMaxEq,DTMinNotEq,DTIn],
    Field(discriminator="op")
]

class _Unset:
    pass

@dataclass
class PlaceHolderValue:
    name: str

class DecisionTable(BaseModel):
    operations: typing.List[RegDTableOps] = Field(default_factory=list)
    operation_inputs: typing.List[str] = Field(default_factory=list)
    outputs: typing.Dict[str, typing.Iterable] = Field(default_factory=dict)
    allow_multi_result: bool = False
    
    def model_post_init(self, __context):
        # Allow previously seen values to not be needed in the execute step
        self._internal_values = {}
        self._default_value = _Unset

    # def __init__(self):
    #     self.mask = None
    #     self.new_index = None
    #     self.outputs = pd.DataFrame()

    def add(self, operation: DecisionTableOp, values: pd.Series, predicate: typing.Iterable, value_name: typing.Optional[str]=None) -> "DecisionTable":
        value_name = get_name(values, value_name)
        self.operations.append(operation(predicate=predicate))
        self.operation_inputs.append(value_name)
        if not isinstance(values, PlaceHolderValue):
            safe_update(self._internal_values, {value_name: values})
        return self
    
    def set_default(self, value):
        self._default_value = value
        return self
    
    def output(self, name: str, value: typing.Iterable) -> "DecisionTable":
        self.outputs[name] = value
        return self
    
    def _get_all_values(self, values):
        values: typing.Dict[str, pd.Series] = dict(values) # make a copy
        safe_update(values, self._internal_values)
        return values
    
    def _lookup_values(self, values):
        assert len(self.operations) >= 1, "At least one operation must be added"

        mask: pd.Series = True
        for op, op_v in zip(self.operations, self.operation_inputs):
            mask = mask & op(values[op_v].values)

        if not self.allow_multi_result and mask.sum(axis=0).max() > 1:
            # msk_idx = np.where(mask.sum(axis=0).max() > 1)
            # TODO better error reporting here
            raise RuntimeError("Found values matching more than one criteria")

        value_idx = np.argmax(mask,axis=0)
        if self._default_value is _Unset:
            assert all(mask[value_idx, range(mask.shape[1])]), "One or more columns didnt match any criteria"
        
        return value_idx, mask

    def execute(self, **values):
        values = self._get_all_values(values)
        outputs = pd.DataFrame(self.outputs)
        value_idx, mask = self._lookup_values(values)
        out_df = outputs.iloc[value_idx].copy()
        if self._default_value is not _Unset:
            default_mask = ~mask[value_idx, range(mask.shape[1])]
            out_df[default_mask] = self._default_value
        return out_df.set_index(values[self.operation_inputs[0]].index)
    
    def trace(self, **values):
        values = self._get_all_values(values)
        value_idx, mask = self._lookup_values(values)

        trace_rows = []
        for idx in value_idx:
            trace = {}
            for op, op_v in zip(self.operations, self.operation_inputs):
                h = trace.get(op_v,{})
                h.update(op.add_trace(idx))
                trace[op_v] = h
            trace_rows.append(trace)
        return trace_rows
