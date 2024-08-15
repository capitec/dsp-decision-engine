import typing

# from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

# class RedefineBehavior:
#     ERROR = "ERROR"
#     WARN = "WARN"
#     IGNORE = "IGNORE"


class _Tree_Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_prefix="SPOCK_TREE_")
    # backend: typing.Optional[str] = None
    # Defines how the tree should behave if a condition or value is defined multiple times with the same name but a different value
    # redefine_behavior: RedefineBehavior = RedefineBehavior.WARN
    max_name_repetition: int = 1_000_000


settings = _Tree_Settings()
