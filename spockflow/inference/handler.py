import typing
from dataclasses import dataclass
from logging import getLogger
from .settings import get_settings
from .exceptions import reraise_common_input_exceptions
from spockflow.inference import util
from functools import partial

if typing.TYPE_CHECKING:
    from spockflow.core import Driver
    from asyncio import Task
    from types import ModuleType
    from .cache import ModelCacheManager
    from spockflow.inference.model_loader import (
        ModelLoader,
        TConfigLoaderMap,
        TModuleList,
        TModuleListMap,
    )
    from spockflow.inference.config.loader.base import ConfigManager, TNamespacedConfig
    from spockflow.inference.io.responses import Response


logger = getLogger(__name__)


def lazy_load_default_model_loader(*args, **kwargs) -> "ModelLoader":
    from spockflow.inference.model_loader import ModelLoader

    return ModelLoader(*args, **kwargs)


def lazy_load_default_model_cache(*args, **kwargs) -> "ModelCacheManager":
    from .cache import ModelCacheManager

    return ModelCacheManager(*args, **kwargs)


def lazy_load_default_model_config(*args, **kwargs) -> "ConfigManager":
    from .config.loader.empty import EmptyConfigManager

    return EmptyConfigManager(*args, **kwargs)


T = typing.TypeVar("T")


@dataclass
class WrappedInputData(typing.Generic[T]):
    data: T
    input_overrides: typing.Optional[typing.Dict[str, typing.Any]] = None
    final_vars: typing.Optional[typing.List[str]] = None


class ServingHandler:
    """
    All items in this class can be overriden. It is possible to inject self by using

    def input_fn(handler, input_data, content_type): ...

    static functions can also be overriden with

    def input_fn(input_data, content_type): ...
    """

    def __init__(self, **kwargs):

        self.entrypoint_conf = kwargs.get("entrypoint_conf")
        self.model_loader_cls = kwargs.get(
            "model_loader_cls", lazy_load_default_model_loader
        )
        self.model_cache_cls = kwargs.get(
            "model_cache_cls", lazy_load_default_model_cache
        )
        self.model_config_cls = kwargs.get(
            "model_config_cls", lazy_load_default_model_config
        )

        self.decoders = kwargs.get("decoders")
        if self.decoders is None:
            from .io.decoders import default_decoders

            self.decoders = default_decoders

        self.encoders = kwargs.get("encoders")
        if self.encoders is None:
            from .io.encoders import default_encoders

            self.encoders = default_encoders

        self._model_cache_refresh_tasks: typing.List["Task"] = []
        self.model_cache: "ModelCacheManager" = None

        for method_name in filter(lambda x: x.endswith("_fn"), dir(self)):
            if not callable(getattr(self, method_name)):
                continue  # Only override functions
            if method_name in kwargs:
                setattr(self, method_name, self._bind_method(kwargs.get(method_name)))

        self.setup_model_cache_fn()

    @classmethod
    def from_module(cls, module: "ModuleType") -> "ServingHandler":
        return cls(**module.__dict__)

    @classmethod
    def from_model_project(cls) -> "ServingHandler":
        from importlib import import_module

        settings = get_settings()
        util.register_model_dir(settings.model_path)
        serving_handler_module = None
        try:
            serving_handler_module = import_module(settings.serving_handler_module)
            # Allow completely custom implementation
            if hasattr(serving_handler_module, "serving_handler_instance"):
                return serving_handler_module.serving_handler_instance
            return cls.from_module(serving_handler_module)
        except ImportError as e:
            logger.warn("Could not import config module. Running with default config.")
        return cls()

    def _bind_method(self, method):
        import types
        import inspect

        # Optionally inject self as first parameter
        method_args = inspect.getfullargspec(method).args
        if method_args[0] == "handler":
            method = types.MethodType(method, self)
        return method

    def model_fn(
        self, model_dir: str
    ) -> typing.Union["TModuleList", "TModuleListMap", "ModuleType"]:
        util.register_model_dir(model_dir)
        entrypoint_conf = self.entrypoint_conf
        if entrypoint_conf is None:
            entrypoint_conf = util.autodiscover_entrypoints(model_dir)
        return util.load_all_entrypoints(entrypoint_conf)

    @staticmethod
    def dag_loader_fn(module_list: "TModuleList", config: dict) -> "Driver":
        """Loads the spock dag given a config"""
        from spockflow.core import Driver

        return Driver(config, *module_list)

    def config_manager_loader_fn(self) -> "TConfigLoaderMap":
        return {"__default__": self.model_config_cls()}

    def _schedule_refresh_latest_model(self, schedule_interval: int):
        if schedule_interval <= 0:
            return
        import asyncio

        loop = asyncio.get_event_loop()

        async def refresh_latest_model():
            nonlocal schedule_interval, asyncio, self
            while True:
                await asyncio.sleep(schedule_interval)
                self.model_cache.refresh_latest_models()

        self._model_cache_refresh_tasks.append(loop.create_task(refresh_latest_model()))

    def setup_model_cache_fn(self):
        settings = get_settings()
        self.model_cache: "ModelCacheManager" = self.model_cache_cls(
            model_loader=self.model_loader_cls(
                model_modules=self.model_fn(settings.model_path),
                config_loaders=self.config_manager_loader_fn(),
                dag_loader_fn=self.dag_loader_fn,
            )
        )

        for model_name in settings.server_pre_init_models:
            self.model_cache.get(model_name)

        if (
            settings.server_pre_init_default_model is None
            and None in self.model_cache.model_loader
        ) or (settings.server_pre_init_default_model):
            self.model_cache.get()

        self._schedule_refresh_latest_model(
            settings.config_cache_latest_reload_interval
        )

    def shutdown_fn(self):
        while self._model_cache_refresh_tasks:
            task = self._model_cache_refresh_tasks.pop()
            task.cancel()

    def input_fn(self, input_data: bytes, content_type: str):
        if content_type not in self.decoders:
            from .exceptions import UnsupportedEncoding

            raise UnsupportedEncoding.from_content_type(
                content_type, list(self.decoders.keys())
            )
        return self.decoders[content_type](input_data)

    @staticmethod
    def pre_process_fn(input_data: typing.Any) -> typing.Dict[str, typing.Any]:
        return input_data

    @staticmethod
    def predict_fn(
        input_data: typing.Dict[str, typing.Any], model: "Driver"
    ) -> typing.Dict[str, typing.Any]:
        return model.raw_execute(
            inputs=input_data,
        )

    @staticmethod
    def post_process_fn(prediction: typing.Dict[str, typing.Any]) -> typing.Any:
        return prediction

    def output_fn(self, prediction: typing.Any, accept: str) -> "Response":
        if accept not in self.encoders:
            # Try a more complex parsing strategy before failing.
            from .io.accept_type_parser import parse_accepted_types

            accepted_types = parse_accepted_types(accept)
            for accept in accepted_types:
                if accept in self.encoders:
                    break
            else:  # For Else not commonly known but this isn't an indentation error
                from .exceptions import UnsupportedAcceptTypeError

                raise UnsupportedAcceptTypeError.from_accept_type(
                    accept, list(self.encoders.keys())
                )
        return self.encoders[accept](prediction)

    @staticmethod
    def override_model_version_fn(data) -> typing.Tuple[typing.Optional[str], str]:
        """
        Allows the user to change the model and version used based on input parameters
        """
        return None, None

    # TODO extend async implementation to here
    def transform_fn(
        self,
        input_data: bytes,  # TODO make this a promise/task
        content_type: str,
        accept: str,
        config_override: typing.Optional["TNamespacedConfig"] = None,
    ) -> "Response":
        log_callback = lambda e: logger.info(
            f"Could not parse input data:\n {input_data}\nWith content type: {content_type}",
            exc_info=e,
        )
        with reraise_common_input_exceptions(log_callback):
            data = self.input_fn(input_data, content_type)
            model_name, model_version = self.override_model_version_fn(data)
            data = self.pre_process_fn(data)

        if config_override is None:
            # TODO Allow this to be awaitable
            model = self.model_cache.get(model_name, model_version)
        else:
            config_log_callback = lambda e: logger.info(
                f"Could not load model with custom config {config_override}", exc_info=e
            )
            with reraise_common_input_exceptions(config_log_callback):
                model = self.model_cache.model_loader.load_model_with_config(
                    model_name, config_override
                )

        with reraise_common_input_exceptions():
            prediction = self.predict_fn(data, model)

        prediction = self.post_process_fn(prediction)
        return self.output_fn(prediction, accept)
