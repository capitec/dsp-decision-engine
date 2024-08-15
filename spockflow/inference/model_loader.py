import typing
from logging import getLogger
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from types import ModuleType
    from spockflow.core import Driver
    from .config.loader.base import ConfigManager


logger = getLogger(__name__)

TModuleList = typing.List["ModuleType"]
TModuleListMap = typing.Dict[str, "TModuleList"]
TConfigLoaderMap = typing.Dict[str, "ConfigManager"]


@dataclass
class VersionedModel:
    model: "Driver"
    version: str
    is_latest: bool = False


# @dataclass
# class ModelWithMetadata:
#     model: "Driver"
#     _model_inputs: typing.Set[str] = None

#     @property
#     def model_inputs(self):
#         if self._model_inputs is None:
#             from .util import get_internal_nodes
#             self._model_inputs = get_internal_nodes(self.model)
#         return self._model_inputs


class ModelLoader:
    def __init__(
        self,
        model_modules: "TModuleListMap",
        config_loaders: "TConfigLoaderMap",
        dag_loader_fn: typing.Callable,
    ):
        self.model_modules: "TModuleListMap" = model_modules
        self.config_loaders: "TConfigLoaderMap" = config_loaders
        self.dag_loader_fn = dag_loader_fn

    def __contains__(self, key: str) -> bool:
        if key is None:
            return "__default__" in self.model_modules
        return key in self.model_modules

    def get_modules_for_model_name(self, model_name):
        modules = self.model_modules.get(
            model_name, self.model_modules.get("__default__")
        )
        if modules is None:
            raise ValueError(
                f'Could not infer the correct modules to use for model: "{model_name}"'
            )
        return modules

    def load_model_with_config(self, model_name, config):
        if model_name is None:
            model_name = "__default__"
        model = self.dag_loader_fn(self.get_modules_for_model_name(model_name), config)
        return model

    def load_model(
        self,
        model_name: str = None,
        model_version: typing.Optional[str] = None,
    ):
        """Construct a versioned model and add it to the cache if not already cached"""
        if model_name is None:
            model_name = "__default__"

        logger.debug(f"Updating cache for {model_name}")
        try:
            config_loader = self.config_loaders.get(
                model_name, self.config_loaders["__default__"]
            )
        except KeyError as e:
            logger.exception(e)
            raise ValueError(
                f'Could not infer the correct config loader to use for model: "{model_name}"'
            ) from e

        is_latest = False
        if model_version is None:
            is_latest = True
            model_version = config_loader.get_latest_version(model_name)
            logger.debug(f"No version specified using latest version = {model_version}")

        # # Letting get return None rather than using contains and get to avoid race condition
        # cached_model = model_cache.get(model_name, model_version)
        # if cached_model is not None:
        #     logger.debug(f"Skipping retrieval of {model_name} {model_version} as it is already cached")
        #     return cached_model

        modules = self.get_modules_for_model_name(model_name)

        try:
            config = config_loader.get_config(model_name, model_version)
        except KeyboardInterrupt as e:
            # Try lessen the blow of a "catch all" except
            raise e from e
        except Exception as e:
            logger.exception(e)
            raise ValueError(
                f'Could not load config for "{model_name}" "{model_version}"'
            ) from e

        model = self.dag_loader_fn(modules, config)

        return VersionedModel(model=model, version=model_version, is_latest=is_latest)
