from typing import Optional, Dict
from .base import ConfigManager, TNamespacedConfig


class EmptyConfigManager(ConfigManager):
    def get_latest_version(self, model_name) -> "str":
        return ""

    def get_config(self, model_name: str, model_version: str) -> TNamespacedConfig:
        return {}

    def save_to_config(
        self,
        model_name: str,
        model_version: str,
        namespace: str,
        config: Dict[str, any],
        key: Optional[str] = None,
    ):
        raise NotImplementedError()
