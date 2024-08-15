from typing import TypeVar, Generic
import pandas as pd
from spockflow.components.tree.v1.core import Tree as Tree

T = TypeVar("T", bound=dict)


class _DataFrameMeta(type):
    def __call__(cls, **kwargs: T) -> pd.DataFrame:
        return pd.DataFrame([kwargs])


class Action(Generic[T], metaclass=_DataFrameMeta):
    """Generic class with a metaclass that returns a pandas DataFrame when called."""

    def __init__(self, **kwargs: T) -> None:
        super().__init__()


__all__ = ["Tree", "Action"]
