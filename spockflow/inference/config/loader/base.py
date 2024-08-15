from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

TNamespacedConfig = Dict[str, Dict[str, Any]]

# TODO consider this might be a bit overkill
# @dataclass(eq=True, order=True, frozen=True)
# class ConfigVersion:
#     major: int = 0
#     minor: int = 0
#     patch: int = 0
#     build_no: int = 0


class ConfigManager(ABC):
    @abstractmethod
    def get_latest_version(self, model_name: str) -> "str": ...

    @abstractmethod
    def get_config(self, model_name: str, model_version: str) -> TNamespacedConfig: ...

    @abstractmethod
    def save_to_config(
        self,
        model_name: str,
        model_version: str,
        namespace: str,
        config: Dict[str, any],
        key: Optional[str] = None,
    ): ...
