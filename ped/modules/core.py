import typing as t
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class BaseModule(BaseModel, ABC):
    cache_kwargs: t.Optional[t.Dict[str, t.Any]] = Field(default_factory=lambda: {"type": "default"})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")
    type: str
    version: t.Optional[str] = Field(default=None, description="Module version, latest if not specified")
    source: t.Optional[str] = Field(default=None, description="Override source for module discovery")
    input_mapping: t.Dict[str, str] = Field(default_factory=dict, description="Map internal names to external names")
    output_mapping: t.Dict[str, str] = Field(default_factory=dict, description="Map internal outputs to external names")
    internal_overrides: t.Dict[str, str] = Field(default_factory=dict, description="Override internal functions")
