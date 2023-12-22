import re
import json
import typing
import asyncio
from io import StringIO
from json import JSONEncoder
from types import ModuleType
from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd
import numpy as np
from logging import getLogger

if typing.TYPE_CHECKING:
    from logging import Logger
    from engine.core import Driver
    from types import ModuleType

logger = getLogger(__name__)

def is_submodule(child: "ModuleType", parent: "ModuleType"):
    return parent.__name__ in child.__name__

def get_config_class_from_module(config_module: "ModuleType"):
    model = getattr(config_module, "model", None) or getattr(config_module, "rule_server", None)
    if model is not None:
        return model
    import inspect
    rule_servers = inspect.getmembers(config_module, lambda x: (
        inspect.isclass(x) and 
        issubclass(x,RuleServer) and
        is_submodule(inspect.getmodule(x), config_module)
    ))
    if len(rule_servers) == 0:
        logger.warn(f"Config module fount but no classes inheriting from {RuleServer.__qualname__} or instance of \"model\" is defined. Reverting to default.")
        return None
    
    rs_name, rs = rule_servers[0]
    if len(rule_servers) > 1:
        raise ValueError(f"Found multiple classes inheriting from rule_server. This may cause unexpected behavior. Please manually instantiate the model as \"model = {rs_name}()\" to explicitly state the config.")
    try:
        return rs()
    except TypeError as e:
        raise ValueError(f"Could not create instance of {rs_name}. Please either manually create instance as \"model = {rs_name}()\" or ensure the model can be created with no arguments.") from e


@dataclass
class ModelResponse:
    content: typing.Any = None
    status_code: int = 200
    headers: typing.Optional[typing.Mapping[str, str]] = None
    media_type: typing.Optional[str] = None


def input_handler(content_type: str, priority:int = 10):
    def inner(fn):
        matcher = re.compile(content_type)
        fn._handles_rule_server_input = priority, matcher.fullmatch
        return fn
    return inner


def output_handler(content_type: str, priority:int = 10):
    def inner(fn):
        matcher = re.compile(content_type)
        fn._handles_rule_server_output = priority, matcher.fullmatch
        fn._encode_output_type = content_type
        return fn
    return inner


class PandasJsonEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, pd.Series):
            return o.to_list()
        elif isinstance(o, pd.DataFrame):
            return {k: v.to_list() for k,v in o.items()}
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.tolist()
        return o.__dict__

@dataclass
class RuleServer:
    root_module: str = "main"
    model: "Driver" = None
    model_config: typing.Dict[str, typing.Any] = field(default_factory=dict)

    def __post_init__(self):
        self._output_handlers = None
        self.init_model()

    @classmethod
    @lru_cache(maxsize=1)
    def _get_input_handlers(cls):
        handlers = sorted([
            (*h, v) 
            for v_name in dir(cls)
            # Not a fan of using walrus but cant make it much neater without
            if (
                not v_name.startswith('_') and
                callable((v:=getattr(cls, v_name))) and 
                (h:=getattr(v,"_handles_rule_server_input",None)) is not None
            )
        ], key=lambda x: x[0])
        return [(does_handle, handler) for _, does_handle, handler in handlers]

    @classmethod
    def _get_output_handlers(cls):
        handlers = sorted([
            (*h, v) 
            for v_name in dir(cls)
            # Not a fan of using walrus but cant make it much neater without
            if (
                not v_name.startswith('_') and
                callable((v:=getattr(cls, v_name))) and 
                (h:=getattr(v,"_handles_rule_server_output",None)) is not None
            )
        ], key=lambda x: x[0])
        return [(does_handle, handler) for _, does_handle, handler in handlers]

    @classmethod
    def _find_first_match(cls, content_type: str, handlers: typing.Tuple[typing.Callable, typing.Callable]):
        for does_match, handler in handlers:
            if does_match(content_type): return handler
        return None
    
    @classmethod
    @lru_cache(maxsize=32)
    def _get_input_handler(cls, content_type: str):
        return cls._find_first_match(content_type, cls._get_input_handlers())
    
    @classmethod
    @lru_cache(maxsize=32)
    def _get_output_handler_for_content_type(cls, content_type: str):
        output_handlers = cls._get_output_handlers()
        if content_type.startswith("*/*"):
            return output_handlers[0][1] if len(output_handlers) > 0 else None
        return cls._find_first_match(content_type, output_handlers)
    
    def _get_output_handler(self, accept_types: typing.Tuple[str, ...]):
        for content_type in accept_types:
            handler = self._get_output_handler_for_content_type(content_type)
            if handler is not None: return handler, content_type
        return None

    def get_modules(self) -> typing.List[ModuleType]:
        """A function to get the core modules used in the engine"""
        from importlib import import_module
        return [import_module(self.root_module)]

    def configure_logger(self, logger: "Logger"):
        """Enable additional configuration on the logger"""
        pass

    def init_model(self):
        """A lifecycle hook that is run after the server has just started to do the heavy-lifting of initializing the model."""
        from engine.core import Driver
        self.model = Driver(self.model_config, *self.get_modules())

    def is_ready(self):
        """A hook to test the health of the model"""
        return self.model is not None
        
    def validate_content_type(self, content_type: str):
        """ An optional lifecycle hook to fail fast. Throw ValueError if not supported. """
        if self._get_input_handler(content_type) is None:
            raise ValueError()
        
    def input_fn(self, data: bytes, content_type: str) -> typing.Union[typing.Dict[str, typing.Any], typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]]:
        """Override this method to change how the function handles inputs. In most cases it is better to declare an input_handler"""
        handler = self._get_input_handler(content_type)
        return handler(self, data, content_type)

    @classmethod
    def _safe_update_dict(cls, value: dict, other:dict):
        key_intersection = set(value.keys()) & other.keys()
        if len(key_intersection) > 0:
            raise ValueError(f"Duplicate keys detected in input data {key_intersection}")
        value.update(other)

    @classmethod
    def get_columns(cls, data: typing.Dict[str, typing.Union["pd.DataFrame", "pd.Series"]]) -> typing.Dict[str, "pd.Series"]:
        res = {}
        for k,v in data.items():
            if isinstance(v, pd.Series):
                cls._safe_update_dict(res, {k:v})
            elif isinstance(v, pd.DataFrame):
                cls._safe_update_dict(res, {f"{k}_{k_col}":v_col for k_col,v_col in v.items()})
            else:
                raise ValueError(f"Get columns only works on Series and DataFrames but found {type(v)}. Please override get_columns to support the additional type.")
        return res
    
    def validate_response_type(self, accept_Types: typing.Tuple[str, ...]):
        """ An optional lifecycle hook to fail fast. Throw ValueError if not supported. """
        if self._get_output_handler(accept_Types) is None:
            raise ValueError()

    def output_fn(self, results: typing.Dict[str, typing.Any], accept_types: typing.Tuple[str, ...]) -> ModelResponse:
        handler, content_type = self._get_output_handler(accept_types)
        res = handler(self, results, content_type)
        output_content_type = content_type
        if content_type == "*/*":
            output_content_type = getattr(handler, "_encode_output_type", None)
        if isinstance(res, str):
            if output_content_type is None:
                raise RuntimeError(f"Could not infer output content type for {handler.__name__}")
            res = ModelResponse(res.encode(), media_type=output_content_type)
        elif isinstance(res, bytes):
            if output_content_type is None:
                raise RuntimeError(f"Could not infer output content type for {handler.__name__}")
            res = ModelResponse(res, media_type=output_content_type)
        else:
            assert isinstance(res, ModelResponse), f"Expected response handler for content type {content_type} to return either a string, bytes or ModelResponse but found {type(res)}"

        return res

    def execute(self, columns: typing.Dict[str, "pd.Series"], **kwargs) -> typing.Dict[str, typing.Any]:
        return self.model.raw_execute(
            inputs=columns,
            **kwargs
        )
    
    @classmethod
    @lru_cache(1)
    def get_json_encoder(cls):
        return PandasJsonEncoder()
    

    @input_handler("application/json", priority=1000)
    def parse_json(self, data: bytes, content_type: str) -> typing.Dict[str, typing.Union["pd.DataFrame", "pd.Series"]]:
        df = pd.DataFrame.from_dict(json.loads(data))
        return {k: v for k,v in df.items()}
    
    @output_handler("application/json", priority=999)
    def encode_json(self, results: typing.Dict[str, typing.Any], accept_type: str) -> ModelResponse:
        return self.get_json_encoder().encode(results)
    
    # @output_handler("text/csv", priority=1000)
    # def encode_csv(self, results: typing.Dict[str, typing.Any], accept_type: str) -> ModelResponse:
    #     res = StringIO()
    #     self.model.adapter.build_result(**results).to_csv(res)
    #     return res.read()

    def run(self, data: bytes, content_type: str, accept_types: typing.Tuple[str, ...]):
        assert self.model is not None, "Model has not been initialized yet"
        data = self.input_fn(data, content_type)
        kwargs = {}
        if isinstance(data, tuple):
            data, kwargs = data
        results = self.execute(data, **kwargs)
        return self.output_fn(results, accept_types)
