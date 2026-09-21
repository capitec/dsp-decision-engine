import typing as t
from uuid import uuid4
from pydantic import BaseModel, field_validator, Field
from collections.abc import Iterable

T = t.TypeVar("T")


class EdgeData(BaseModel):
    sourceIndex: int


class MultiEdgeData(BaseModel):
    sourceIndex: t.List[int]

    @field_validator("sourceIndex", mode="before")
    @classmethod
    def ensure_list(cls, v: t.Any) -> t.List[int]:
        if not isinstance(v, Iterable) or isinstance(v, str):
            return [v]
        return list(v)


class GenericEdge(BaseModel, t.Generic[T]):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    data: T

    @field_validator("id", mode="before")
    @classmethod
    def ensure_id(cls, v: t.Any) -> str:
        return str(uuid4()) if v is None else v


MultiSourceEdge = GenericEdge[MultiEdgeData]
Edge = GenericEdge[EdgeData]
