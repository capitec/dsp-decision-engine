import typing as t
from abc import abstractmethod

import polars as pl

from decider.types import TInputType, TOutputType
from decider.modules.core import BaseModule, BaseExecuteModule


if t.TYPE_CHECKING:
    from decider.executor import Executor


FrameInput = t.Union[str, BaseModule]


def _resolve_frame(
    ref: FrameInput,
    inputs: TInputType,
    executor: "Executor",
) -> pl.LazyFrame:
    if isinstance(ref, str):
        frame = inputs[ref]
        return frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    result = ref(inputs, executor=executor)
    return result.lazy() if isinstance(result, pl.DataFrame) else result


class FrameRef(BaseExecuteModule):
    """Extracts a named frame from inputs as 'input', enabling frame routing.

    Used to feed a specific named input into a sub-pipeline:
        FrameRef("input1") | scorer  →  routes input1 through scorer
    """

    type: t.Literal["frame_ref"]

    def execute(self, inputs: TInputType, executor: "Executor") -> TOutputType:
        frame = inputs[self.name]
        return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


class FrameModule(BaseExecuteModule):
    """Module that combines named input frames and returns a new LazyFrame."""

    @abstractmethod
    def execute(self, inputs: TInputType, executor: "Executor") -> TOutputType:
        ...


class JoinModule(FrameModule):
    type: t.Literal["join"]
    left: FrameInput
    right: FrameInput
    on: t.Union[str, t.List[str]]
    how: str = "left"

    model_config = {"arbitrary_types_allowed": True}

    def _compute_input_frame_keys(self) -> t.List[str]:
        keys = []
        if isinstance(self.left, str):
            keys.append(self.left)
        else:
            keys.extend(self.left.get_input_frame_keys())
        if isinstance(self.right, str):
            keys.append(self.right)
        else:
            keys.extend(self.right.get_input_frame_keys())
        return list(dict.fromkeys(keys))  # deduplicate, preserve order

    def execute(self, inputs: TInputType, executor: "Executor") -> TOutputType:
        left_frame = _resolve_frame(self.left, inputs, executor)
        right_frame = _resolve_frame(self.right, inputs, executor)
        return left_frame.join(right_frame, on=self.on, how=self.how)
