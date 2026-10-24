import functools
import os
import typing as t
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_workers() -> int:
    """nproc * 2 + 1 — sensible default for I/O-bound async workers."""
    return os.cpu_count() * 2 + 1


class ServeSettings(BaseModel):
    """Settings for the Decikit HTTP server."""
    host: str = "0.0.0.0"
    port: int = 8080
    # None means use _default_workers() at serve time so nproc is evaluated
    # on the target machine, not at settings-parse time.
    workers: t.Optional[int] = None
    server: t.Literal["starlette", "sanic"] = "starlette"


class EngineSettings(BaseModel):
    """The engine serving binds with: its `type` tag plus that engine's own fields (e.g. `strict_compile`)."""
    model_config = ConfigDict(extra='allow')
    type: str = "fused"


class APISettings(BaseModel):
    """What the server serves: `handler` is an import path; `pipeline` is an import path or a
    `.yaml`/`.yml`/`.json` flow file, both resolved from `code_path`."""
    # Forbidden so a retired setting (`DECIKIT_API__MODE`) fails loading instead of being ignored.
    model_config = ConfigDict(extra='forbid')
    code_path: str = "."
    pipeline: str = "pipeline:build"
    handler: str = "inference:Handler"
    engine: EngineSettings = Field(default_factory=EngineSettings)


class BuildSettings(BaseModel):
    """What `decikit build` and `RequestHandler.stage()` check."""
    # ShadowedTableWarning: a params document sets a table whose default rows come from a config document.
    warn_shadowed_tables: bool = True


class DecikitConfigSettings(BaseModel):
    """The config store to build: its `type` tag plus that store's own fields (e.g. `basepath`)."""
    model_config = ConfigDict(extra='allow')
    type: str = "file:json"

    def get(self):
        from decikit.config import ConfigStore
        return ConfigStore.resolve(self.type).model_validate(self.model_dump())


class DecikitSettings(BaseSettings):
    """Main settings for the Decikit application."""

    model_config = SettingsConfigDict(
        env_prefix="Decikit_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    serve: ServeSettings = Field(default_factory=ServeSettings)
    api: APISettings = Field(default_factory=APISettings)
    build: BuildSettings = Field(default_factory=BuildSettings)
    config: DecikitConfigSettings = Field(default_factory=DecikitConfigSettings)
    datasets_path: str = "${PROJECT_ROOT}/datasets"  # folder of named datasets; `${PROJECT_ROOT}` is the nearest ancestor holding `configs/`
    checks_path: str = "${PROJECT_ROOT}/checks.yaml"  # the project's checks file; resolved like `datasets_path`


@functools.cache
def get_settings() -> DecikitSettings:
    """The process's settings, read from the environment on first use.

    `get_settings.cache_clear()` makes the next call read the environment again.

    Example::

        get_settings().serve.port  # 8080
    """
    return DecikitSettings()

