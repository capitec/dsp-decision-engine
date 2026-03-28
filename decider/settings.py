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

class DeciderSettings(BaseSettings):
    """Main settings for the Decider application."""

    model_config = SettingsConfigDict(
        env_prefix="Decider_",
        case_sensitive=False,
    )

    ext: DeciderAppExtensionSettings = Field(default_factory=DeciderAppExtensionSettings)
    api: APISettings = Field(default_factory=APISettings)


settings = DeciderSettings()