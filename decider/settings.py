import typing as t
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class DeciderSettings(BaseSettings):
    """Main settings for the Decider application."""

    model_config = SettingsConfigDict(
        env_prefix="Decider_",
        case_sensitive=False,
    )

    ext: DeciderAppExtensionSettings = Field(default_factory=DeciderAppExtensionSettings)
    api: APISettings = Field(default_factory=APISettings)
    config_poll_duration_s: int = SETTINGS_DEFAULT_CONFIG_POLL_DURATION_S


settings = DeciderSettings()


# ========================================
# Executor Configuration
# ========================================

if t.TYPE_CHECKING:
    from decider.executor import Executor


_default_executor: t.Optional["Executor"] = None


def get_default_executor() -> "Executor":
    """Get the default executor from settings.

    If no executor has been set, creates and returns a SimpleExecutor.

    Returns:
        The default executor instance

    Example:
        >>> executor = get_default_executor()
        >>> compiled = module.compile(executor)
    """
    global _default_executor
    if _default_executor is None:
        from decider.executor import SimpleExecutor
        _default_executor = SimpleExecutor()
    return _default_executor


def set_default_executor(executor: "Executor") -> None:
    """Set the default executor globally.

    This affects all modules that don't specify an explicit executor.

    Args:
        executor: The executor instance to use as default

    Example:
        >>> from decider.executor import WaveExecutor
        >>> set_default_executor(WaveExecutor())
        >>> # All modules now use WaveExecutor by default
    """
    global _default_executor
    _default_executor = executor


def reset_default_executor() -> None:
    """Reset the default executor to None (will be lazy-initialized on next use)."""
    global _default_executor
    _default_executor = None