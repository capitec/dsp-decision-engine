import os
import typing

from typing_extensions import Annotated

from functools import lru_cache
from pydantic import Field
from pydantic.functional_validators import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def semicolon_separated(v: str) -> typing.List[str]:
    assert isinstance(v, str), "must be a semi-colon separated string"
    return [iv.split(",") for iv in v.split(";")]


def get_cpu_count():
    import multiprocessing

    return int(multiprocessing.cpu_count())


class _Runtime_Settings(BaseSettings):
    # This defines how the config is loaded from environment variables
    # e.g prefix can be changed by setting os.environ['MODEL_PREFIX']
    model_config = SettingsConfigDict(case_sensitive=False, env_prefix="MODEL_")

    # These two variables allow the directory of the model to be changes
    prefix: str = "/opt/ml/"
    relative_path: str = "model"
    relative_requirements_path: str = "requirements.txt"

    serving_handler_module: str = "inference"

    # This is a list to describe which python file contains the dag to be executed
    default_entrypoints: typing.Annotated[
        typing.List[typing.List[str]], BeforeValidator(semicolon_separated)
    ] = "main;run;proc"

    proc_prefix: str = "proc_"
    config_cache_capacity: int = 8
    config_cache_latest_ttl: float = float("inf")
    config_cache_latest_reload_interval: float = 0

    server_timeout: int = 60
    server_workers: int = Field(default_factory=get_cpu_count)
    server_inference_with_config_endpoint: bool = True
    server_include_visualize: bool = True
    # Allow api to override model values using inputs
    server_enable_model_overrides: bool = True
    server_model_output_override_header: typing.Optional[str] = "x-model-outputs"
    server_model_name_header: typing.Optional[str] = "x-model-name"
    server_model_version_header: typing.Optional[str] = "x-model-version"

    server_pre_init_models: list = Field(default_factory=list)
    # None will try to init the model if it exists
    # True will init the model and raise an error if it doesnt exist
    server_pre_init_default_model: typing.Optional[bool] = None
    # If the model should be created on startup or on first request
    server_init_on_startup: bool = True

    @property
    def model_path(self):
        return os.path.join(self.prefix, self.relative_path)

    @property
    def requirements_path(self):
        return os.path.join(self.model_path, self.relative_requirements_path)


@lru_cache(maxsize=1)
def get_settings():
    return _Runtime_Settings()


__all__ = [get_settings]
