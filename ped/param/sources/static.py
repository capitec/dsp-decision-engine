import typing as t
from .core import BaseSource
from ._ext import register_source


class StaticSource(BaseSource):
    type: t.Literal['static'] = "static"
    values: dict[str, t.Any]

    def requires_refresh(self, **kwargs) -> bool:
        return False

    def get(self, key: str, **kwargs) -> t.Any:
        return self.values[key]


register_source(StaticSource)
