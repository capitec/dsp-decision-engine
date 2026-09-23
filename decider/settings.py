import os
import typing as t
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_workers() -> int:
    """nproc * 2 + 1 — sensible default for I/O-bound async workers."""
    return os.cpu_count() * 2 + 1


class ServeSettings(BaseModel):
    """Settings for the Decider HTTP server."""
    host: str = "0.0.0.0"
    port: int = 8080
    # None means use _default_workers() at serve time so nproc is evaluated
    # on the target machine, not at settings-parse time.
    workers: t.Optional[int] = None


class APISettings(BaseModel):
    """What the server serves: `pipeline` and `handler` are import paths, resolved from `code_path`."""
    code_path: str = "."
    pipeline: str = "pipeline:build"
    handler: str = "inference:Handler"
    mode: str = "fused"


class DeciderConfigSettings(BaseModel):
    """The config store to build: its `type` tag plus that store's own fields (e.g. `basepath`)."""
    model_config = ConfigDict(extra='allow')
    type: str = "file:json"

    def get(self):
        from decider.config import ConfigStore
        return ConfigStore.resolve(self.type).model_validate(self.model_dump())


class DeciderSettings(BaseSettings):
    """Main settings for the Decider application."""

    model_config = SettingsConfigDict(
        env_prefix="Decider_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    serve: ServeSettings = Field(default_factory=ServeSettings)
    api: APISettings = Field(default_factory=APISettings)
    config: DeciderConfigSettings = Field(default_factory=DeciderConfigSettings)


settings = DeciderSettings()

