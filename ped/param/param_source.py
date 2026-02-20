import typing as t
from dataclasses import dataclass

from ped.types import TInputType
from .sources import VersionedSource



# Compatibility for Python < 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass




@dataclass
class ParameterSourceProvider:
    sources: dict[str, VersionedSource]
    inputs: TInputType
    
    def resolve(
        self, 
        source_name: str, key: str, args: t.Tuple[str, ...]
    ) -> t.Any:
        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not found in provider.")

        return self.sources[source_name].get(
            key, 
            inputs=self.inputs,
            args=args
        )
