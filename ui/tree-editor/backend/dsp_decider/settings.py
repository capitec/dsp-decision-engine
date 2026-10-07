import json
import typing as t
from logging import getLogger
from pathlib import Path
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)
from dsp_decider.auth.settings import AuthenticationSettings
from dsp_decider.cache.settings import TCacheProviders, InMemoryProvider
from dsp_decider.db.settings import DatabaseSettings
from dsp_decider.data_sources.settings import DataSourcesSettings

logger = getLogger(__name__)


SETTINGS_PATHS = [
    "~/.config/dsp-decider/config.json",
    "~/.config/dsp-decider/config.yaml",
    "~/.config/dsp-decider/config.yml",
    "/etc/dsp-decider/config.json",
    "/etc/dsp-decider/config.yaml",
    "/etc/dsp-decider/config.yml",
    "./config.json",
    "./config.yaml",
    "./config.yml",
]


class ConfigFileSource(PydanticBaseSettingsSource):
    """
    A settings source that loads from JSON/YAML config files.
    Ignores missing files, but raises errors if files exist but can't be loaded.
    """

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        self.config_data = self._load_config()

    def _load_config(self) -> t.Dict[str, t.Any]:
        """Load config from first available file"""
        for path in SETTINGS_PATHS:
            expanded_path = Path(path).expanduser()
            if not expanded_path.is_file():
                continue

            try:
                logger.info(f"Loading base config from {expanded_path}")
                if path.endswith(".json"):
                    return json.loads(expanded_path.read_text())
                elif path.endswith((".yaml", ".yml")):
                    import yaml

                    return yaml.safe_load(expanded_path.read_text())
            except Exception as e:
                # File exists but can't be loaded - this is an error
                raise ValueError(
                    f"Failed to load config from {expanded_path}: {e}"
                ) from e

        # No config files found - this is OK
        logger.info("No config files found, using environment variables only")
        return {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[t.Any, str, bool]:
        """Get field value from loaded config data"""
        field_value = self.config_data.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: t.Any, value_is_complex: bool
    ) -> t.Any:
        """Prepare field value - just return as-is"""
        return value

    def __call__(self) -> t.Dict[str, t.Any]:
        """Return all config data for this source"""
        d: t.Dict[str, t.Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            if field_value is not None:
                field_value = self.prepare_field_value(
                    field_name, field, field_value, value_is_complex
                )
                d[field_key] = field_value

        return d


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False, env_prefix="DSP_", env_nested_delimiter="__"
    )

    auth: AuthenticationSettings = Field(default_factory=AuthenticationSettings)
    cache: TCacheProviders = Field(default_factory=InMemoryProvider)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    datasources: DataSourcesSettings = Field(default_factory=DataSourcesSettings)
    ui_build_path: t.Optional[str] = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources priority:
        1. init_settings (direct instantiation)
        2. ConfigFileSource (config files)
        3. env_settings (environment variables) - OVERRIDES config file
        4. file_secret_settings (Docker secrets, etc.)
        """
        return (
            init_settings,
            ConfigFileSource(settings_cls),
            env_settings,
            file_secret_settings,
        )

    # Legacy properties for backward compatibility
    @property
    def DATABASE_URL(self) -> str:
        return self.db.database_url

    @property
    def DEBUG(self) -> bool:
        return self.db.echo


# Create settings instance with layered configuration:
# 1. Config files (base)
# 2. Environment variables (override)
settings = AppSettings()
