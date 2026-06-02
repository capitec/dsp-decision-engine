from dataclasses import dataclass
import asyncio
import importlib.util
import typing as t

import polars as pl

from decider.config import ConfigManager
from decider.config.versioned import Version
from decider.modules import GraphModule
import decider.exceptions as exc
from .parse import DEFAULT_INPUT_HANDLERS, ParserConfig
from .format import DEFAULT_OUTPUT_FORMATTERS, Response
from .media_types import MediaType


@dataclass
class RequestHandler:
    config_manager: "ConfigManager"
    root_module: str = "main"
    _update_task: asyncio.Task = None
    _constructed_module: GraphModule = None
    _constructed_version: Version = None
    _constructed_parse_config: t.Optional[ParserConfig] = None

    async def init_fn(self):
        await self.config_manager.get_latest()
        self._update_task = asyncio.create_task(self.config_manager.subscribe_version_updates())


    def module_fn(self) -> t.Tuple[GraphModule, ParserConfig]:
        try:
            with self.config_manager.current_version_context() as versioned_config:
                if self._constructed_version is not None and versioned_config.version == self._constructed_version:
                    return self._constructed_module, self._constructed_parse_config

                module_config = versioned_config.config.get(self.root_module)
                if module_config is None:
                    raise ValueError(f"No config found for root module '{self.root_module}' in the current versioned config.")
                self._constructed_module = GraphModule.model_validate(module_config).root
                self._constructed_version = versioned_config.version
                self._constructed_parse_config = ParserConfig(input_frame_keys=self._constructed_module.get_input_frame_keys())
                return self._constructed_module, self._constructed_parse_config

        except exc.BaseConfigurationError:
            raise
        except ValueError as e:
            raise exc.ModuleLoadError.from_value_error(e)
        except Exception as e:
            raise exc.ModuleLoadError(str(e))
        
    async def input_fn(self, data: bytes, content_type: str, parse_config: t.Optional[ParserConfig] = None):
        handler = DEFAULT_INPUT_HANDLERS.get(content_type)
        if handler is None:
            raise exc.UnsupportedContentTypeError(f"Unsupported content type: {content_type!r}")
        try:
            return await handler(data, parse_config or ParserConfig())
        except Exception as e:
            raise exc.InputParsingError(str(e))
        
    def output_fn(self, output: pl.DataFrame, accept: str) -> Response:
        formatter = DEFAULT_OUTPUT_FORMATTERS.get(accept)
        if formatter is None:
            raise exc.UnsupportedAcceptError(f"Unsupported Accept type: {accept!r}")
        try:
            response = formatter(output)
            if response.media_type is None:
                if accept == MediaType.ANY.value:
                    raise exc.DeciderRuntimeError("Configured Format for MediaType.ANY must return a Response with a specific media_type, got None")
                response = Response(content=response.content, media_type=accept)
            return response
        except exc.DeciderError:
            raise
        except Exception as e:
            raise exc.OutputFormattingError(str(e))

    async def process_fn(self, data: bytes, accept: str, content_type: str) -> Response:
        module, parse_config = self.module_fn()
        input_data = await self.input_fn(data, content_type, parse_config)
        result_df = module(input_data)
        return self.output_fn(result_df, accept)

    async def shutdown_fn(self):
        self._update_task.cancel()
        try:
            await self._update_task
        except asyncio.CancelledError:
            pass


def construct_handler_from_settings() -> RequestHandler:
    import sys
    import os
    from decider.settings import settings

    config_manager = settings.config.get()

    handler_path: str = getattr(settings.api, "handler", "inference:Handler")
    module_name, _, class_name = handler_path.partition(":")

    handler_constructor = None

    # Try to load a custom handler class from the working directory
    module_file = os.path.join(os.getcwd(), f"{module_name}.py")
    if os.path.exists(module_file):
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # raises if file has errors
        handler_constructor = getattr(mod, class_name, None)

    if handler_constructor is None:
        handler_constructor = RequestHandler

    return handler_constructor(config_manager=config_manager)