import typing
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from dataclasses import dataclass
from spockflow.nodes import VariableNode, creates_node
from spockflow._serializable import Series, DataFrame

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
        criteria_mask = values[None] >= self.pred[:, None]
        criteria_mask[np.isnan(self.pred)] = True
        return criteria_mask

    def add_trace(self, idx):
        return {"lower": self.pred[idx]}


class DTMax(DecisionTableOp):
    op: typing.Literal["MAX"] = "MAX"

    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] < self.pred[:, None]
        criteria_mask[np.isnan(self.pred)] = (
            True  # TODO verify that this works for isnull
        )
        return criteria_mask

    def add_trace(self, idx):
        return {"upper": self.pred[idx]}


class DTMinNotEq(DecisionTableOp):
    """DTMIN is the default of min inclusive this is defined to be not inclusive of the lower bound."""

    op: typing.Literal["MINNEQ"] = "MINNEQ"

    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] > self.pred[:, None]
        criteria_mask[np.isnan(self.pred)] = True
        return criteria_mask

    def add_trace(self, idx):
        return {"lower": self.pred[idx]}


class DTMaxEq(DecisionTableOp):
    op: typing.Literal["MAXEQ"] = "MAXEQ"

    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        criteria_mask = values[None] <= self.pred[:, None]
        criteria_mask[np.isnan(self.pred)] = (
            True  # TODO verify that this works for isnull
        )
        return criteria_mask

    def add_trace(self, idx):
        return {"upper": self.pred[idx]}


class DTIn(DecisionTableOp):
    op: typing.Literal["IN"] = "IN"

    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        pred = self.pred
        return np.apply_along_axis(  # TODO maybe write this as a numba function
            lambda v: (
                np.isin(values, eval(v[0]))
                if not pd.isnull(v[0])
                else [True] * len(values)
            ),
            0,
            pred[None],
        )

    def add_trace(self, idx):
        return {"in": self.pred[idx]}


class DTEq(DecisionTableOp):
    op: typing.Literal["EQ"] = "EQ"
    tol: typing.Optional[float] = 1e-9

    def __call__(self, values: ArrayV) -> ArrayPxV[np.bool_]:
        if self.tol is not None:  # Set as a default
            criteria_mask = np.abs(values[None] - self.pred[:, None]) < self.tol
        else:
            criteria_mask = values[None] == self.pred[:, None]
        criteria_mask[np.isnan(self.pred)] = True
        return criteria_mask

    def add_trace(self, idx):
        return {"equals": self.pred[idx]}


RegDTableOps = typing.Annotated[
    typing.Union[DTMin, DTMax, DTMaxEq, DTMinNotEq, DTIn], Field(discriminator="op")
]


@dataclass
class LookupResult:
    value_idx: np.ndarray
    mask: pd.Series


class DecisionTable(VariableNode):
    operations: typing.List[RegDTableOps] = Field(default_factory=list)
    operation_inputs: typing.List[str] = Field(default_factory=list)
    outputs: typing.Dict[str, typing.Iterable] = Field(default_factory=dict)
    allow_multi_result: bool = False
    default_value: typing.Union[Series, DataFrame, None] = None
    # Allow previously seen values to not be needed in the execute step
    _internal_values: typing.Dict[str, Series] = PrivateAttr(default_factory=dict)

    def add(
        self,
        operation: DecisionTableOp,
        values: pd.Series | str,
        predicate: typing.Iterable,
        value_name: typing.Optional[str] = None,
    ) -> "DecisionTable":
        values_is_placeholder = isinstance(values, str)
        if value_name is None and values_is_placeholder:
            value_name = values
        value_name = get_name(values, value_name)
        self.operations.append(operation(predicate=predicate))
        self.operation_inputs.append(value_name)
        if not values_is_placeholder:
            safe_update(self._internal_values, {value_name: values})
        return self

    def set_default(self, value):
        self.default_value = value
        return self

    def output(self, name: str, value: typing.Iterable) -> "DecisionTable":
        self.outputs[name] = value
        return self

    def _get_inputs(self, fn):
        return {
            k: pd.Series
            for k in self.operation_inputs
            if k not in self._internal_values
        }

    @creates_node(kwarg_input_generator="_get_inputs")
    def all_values(
        self, **kwargs: typing.Dict[str, pd.Series]
    ) -> typing.Dict[str, pd.Series]:
        values: typing.Dict[str, pd.Series] = dict(kwargs)  # make a copy
        safe_update(values, self._internal_values)
        return values

    @creates_node()
    def lookup_values(self, all_values: typing.Dict[str, pd.Series]) -> LookupResult:
        assert len(self.operations) >= 1, "At least one operation must be added"

        mask: pd.Series = True
        for op, op_v in zip(self.operations, self.operation_inputs):
            mask = mask & op(all_values[op_v].values)

        if not self.allow_multi_result and mask.sum(axis=0).max() > 1:
            # msk_idx = np.where(mask.sum(axis=0).max() > 1)
            # TODO better error reporting here
            raise RuntimeError("Found values matching more than one criteria")

        value_idx = np.argmax(mask, axis=0)
        if self.default_value is None:
            assert all(
                mask[value_idx, range(mask.shape[1])]
            ), "One or more columns didnt match any criteria"

        return LookupResult(value_idx, mask)

    @creates_node(is_namespaced=False)
    def get_outputs(
        self, lookup_values: LookupResult, all_values: typing.Dict[str, pd.Series]
    ) -> pd.DataFrame:
        value_idx, mask = lookup_values.value_idx, lookup_values.mask
        outputs = pd.DataFrame(self.outputs)
        out_df = outputs.iloc[value_idx].copy()
        if self.default_value is not None:
            default_mask = ~mask[value_idx, range(mask.shape[1])]
            out_df[default_mask] = self.default_value
        return out_df.set_index(all_values[self.operation_inputs[0]].index)

    @creates_node()
    def trace(self, lookup_values: LookupResult) -> dict:  # TODO improve type-hinting
        value_idx, mask = lookup_values.value_idx, lookup_values.mask
        trace_rows = []
        for idx in value_idx:
            trace = {}
            for op, op_v in zip(self.operations, self.operation_inputs):
                h = trace.get(op_v, {})
                h.update(op.add_trace(idx))
                trace[op_v] = h
            trace_rows.append(trace)
        return trace_rows
