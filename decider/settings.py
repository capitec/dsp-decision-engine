import typing as t
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache


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

SETTINGS_DEFAULT_CONFIG_POLL_DURATION_S: int = 10

class DeciderConfigSettings(BaseModel):
    model_config = ConfigDict(extra='allow')
    type: str = "file:json"

    def get(self):
        from decider.config import ConfigManager
        return ConfigManager.model_validate(self.model_dump())


class DeciderSettings(BaseSettings):
    """Main settings for the Decider application."""

    model_config = SettingsConfigDict(
        env_prefix="Decider_",
        case_sensitive=False,
    )

    ext: DeciderAppExtensionSettings = Field(default_factory=DeciderAppExtensionSettings)
    api: APISettings = Field(default_factory=APISettings)
    config: DeciderConfigSettings = Field(default_factory=DeciderConfigSettings)


settings = DeciderSettings()


# ========================================
# Executor Configuration
# ========================================

if t.TYPE_CHECKING:
    from decider.executor import Executor



@cache
def get_default_executor() -> "Executor":
    """Get the default executor from settings.

    If no executor has been set, creates and returns a SimpleExecutor.

    Returns:
        The default executor instance

    Example:
        >>> executor = get_default_executor()
        >>> compiled = module.compile(executor)
    """
    from decider.executor import SimpleExecutor
    return SimpleExecutor()
