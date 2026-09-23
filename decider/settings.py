import os
import typing as t
from pydantic import BaseModel, Field, field_validator, ConfigDict
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
    """Settings for the Decider API."""
    root_path: str = "./model/code"
    flow_subpath: str = ""
    init_module: t.Optional[str] = "inference"

class DeciderAppExtensionSettings(BaseModel):
    """Settings for the Decider application extensions."""
    extension_path: str = "decider_extensions"
    extension_imports: t.List[str] = []

    @field_validator("extension_imports", mode="before")
    @classmethod
    def split_comma_separated(cls, v: t.Any) -> t.Any:
        """Allow comma-separated values from environment variables."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

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
    ext: DeciderAppExtensionSettings = Field(default_factory=DeciderAppExtensionSettings)
    api: APISettings = Field(default_factory=APISettings)
    config: DeciderConfigSettings = Field(default_factory=DeciderConfigSettings)


settings = DeciderSettings()

