from spockflow._imports import assert_module_exists

assert_module_exists("yaml")
from yaml import dump, load

try:
    from yaml import CDumper as Dumper, CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Dumper, Loader
import os
from .base import ConfigManager, TNamespacedConfig
from pydantic_settings import BaseSettings, SettingsConfigDict


class YamlConfigManager(ConfigManager, BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False, env_prefix="CONFIG_MANAGER_"
    )

    config_path: str = os.path.join(".", "config")

    def model_path(self, model_name: str) -> str:
        return os.path.join(self.config_path, model_name)

    def get_latest_version(self, model_name: str) -> "str":
        if model_name == "__default__":
            paths = os.listdir(self.config_path)
            if model_name not in paths:
                assert (
                    len(paths) == 1
                ), "can only use default when there is one model or an explicit __default__"
                model_name = paths[0]
        return sorted(os.listdir(self.model_path(model_name)))[-1]

    def get_config(self, model_name: str, model_version: str) -> TNamespacedConfig:
        from glob import glob

        r = {}
        for f in glob(
            os.path.join(self.model_path(model_name), model_version, "*.yml")
        ):
            ns = os.path.splitext(os.path.split(f)[1])[0]
            with open(f) as fp:
                r[ns] = load(fp, Loader=Loader)
        return r

    def save_to_config(
        self,
        model_name: str,
        model_version: str,
        namespace: str,
        config: TNamespacedConfig,
        key: str | None = None,
    ):
        save_path = (
            os.path.join(self.model_path(model_name), model_version, namespace) + ".yml"
        )
        if key is not None:
            if os.path.isfile(save_path):
                with open(save_path) as fp:
                    curr_config = load(fp, Loader=Loader)
                curr_config[key] = config
                config = curr_config
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        with open(save_path, "w") as fp:
            dump(config, fp, Dumper=Dumper)
