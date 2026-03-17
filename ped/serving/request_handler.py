import sys
from io import BytesIO
import typing as t
from types import ModuleType
from dataclasses import dataclass, field
import polars as pl
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

if t.TYPE_CHECKING:
    from ped.flow import FlowConfiguration


async def initialize_request_handler():
    from pathlib import Path
    from ped.settings import settings
    from ped import initialize_ped


    root_path = Path(settings.api.root_path)
    sys.path.insert(0, str(root_path.resolve()))
    # Must maybe think about this a bit however i think the ped_extensions should be relative to the api root path
    settings.ext.extension_path = str((root_path / settings.ext.extension_path).resolve())
    # Load all extensions
    initialize_ped()

    # The below can import from this module and update the handler with custom logic
    if settings.api.init_module:
        init_file = root_path / (settings.api.init_module+".py")
        if init_file.exists():
            from importlib import import_module
            mod = import_module(settings.api.init_module)
            handler = RequestHandler.from_module(mod)
    if handler is None:
        handler = RequestHandler()

    await handler.init_fn()

    return handler


class DefaultParsedInputTypeDict(t.TypedDict):
    data: t.List[dict] | pl.DataFrame
    parameters: t.NotRequired[dict]
    config: t.NotRequired[dict]
DefaultParsedInputType = t.Union[DefaultParsedInputTypeDict, t.List[dict]]

if sys.version_info >= (3, 13):
    TParsedInputType = t.TypeVar("TParsedInputType", default=DefaultParsedInputType)
else:
    TParsedInputType = t.TypeVar("TParsedInputType", bound=DefaultParsedInputType)


async def parse_application_json(request: "Request") -> dict:
    return await request.json()

async def parse_application_jsonl(request: "Request") -> dict:
    return {"data": pl.read_ndjson(request.stream())}

async def parse_application_x_parquet(request: "Request") -> dict:
    return {"data": pl.read_parquet(request.stream())}

async def parse_text_csv(request: "Request") -> dict:
    return {"data": pl.read_csv(request.stream())}

async def parse_application_excel(request: "Request") -> dict:
    return {"data": pl.read_excel(request.stream())}

def format_application_json(result: pl.DataFrame) -> "Response":
    # Most clients send 1 in and expect 1 out so use that as the default behavior
    if len(result) == 1:
        return JSONResponse(content=result[0].to_dict())
    else:
        return JSONResponse(content=result.to_dicts())

def format_application_jsonl(result: pl.DataFrame) -> "Response":
    f = BytesIO()
    result.write_ndjson(f)
    f.seek(0)
    return Response(
        content=f.read(), 
        media_type="application/jsonl"
    )

def format_application_x_parquet(result: pl.DataFrame) -> "Response":
    f = BytesIO()
    result.write_parquet(f)
    f.seek(0)
    return Response(
        content=f.read(), 
        media_type="application/x-parquet"
    )

def format_text_csv(result: pl.DataFrame) -> "Response":
    f = BytesIO()
    result.write_csv(f)
    f.seek(0)
    return Response(
        content=f.read(), 
        media_type="text/csv"
    )

@dataclass
class RequestHandler(t.Generic[TParsedInputType]):
    flow_catalog: t.Dict[str, "FlowConfiguration"] = None
    default_flow: str = "default"

    _DEFAULT_INPUT_HANDLERS = {
        "application/json": parse_application_json,
        "application/jsonl": parse_application_jsonl,
        "application/x-parquet": parse_application_x_parquet,
        "text/csv": parse_text_csv,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": parse_application_excel,
        "application/vnd.ms-excel": parse_application_excel,
    }

    _DEFAULT_RESULT_FORMATTERS = {
        "*/*": format_application_json,  # Default to JSON for any Accept header
        "application/json": format_application_json,
        "application/jsonl": format_application_jsonl,
        "application/x-parquet": format_application_x_parquet,
        "text/csv": format_text_csv,
    }

    @classmethod
    def from_module(cls, module: ModuleType) -> "RequestHandler":
        """Create handler from module, auto-discovering overridable methods"""
        overrides = {}
        ret = cls()
        for method_name in cls.OVERRIDABLE_METHODS:
            if (m := getattr(module, method_name, None)):
                setattr(ret, method_name, ret._bind_method(m))
        return ret

    def _bind_method(self, method):
        """Bind external method, optionally injecting handler as first param"""
        import types
        import inspect
        
        method_args = inspect.getfullargspec(method).args
        if method_args and method_args[0] == "handler":
            method = types.MethodType(method, self)
        return method

    async def init_fn(self):
        """Load flow configurations from the configured extension path and return a mapping of flow name to configuration."""
        from ped.settings import settings
        from pathlib import Path
        from ped.flow import FlowConfiguration

        flow_path = Path(settings.api.root_path) / settings.api.flow_subpath
        flow_configs = {}
        for flow_file in flow_path.glob("*.json"):
            flow_name = flow_file.stem
            flow_configs[flow_name] = FlowConfiguration.model_validate(flow_file)

        self.flow_catalog = flow_configs

    def select_flow_fn(self, parsed_inputs: TParsedInputType) -> "FlowConfiguration":
        """Select a flow configuration based on the incoming request. By default, this returns the flow named 'default'."""
        if self.flow_catalog is None:
            raise ValueError("Flow catalog not loaded")
        
        if isinstance(parsed_inputs, dict) and "config" in parsed_inputs:
            flow_name = parsed_inputs["config"].get("flow", self.default_flow)
            if flow_name in self.flow_catalog:
                return self.flow_catalog[flow_name]

        if self.default_flow not in self.flow_catalog:
            raise ValueError(f"Default flow '{self.default_flow}' not found in flow catalog")

        return self.flow_catalog[self.default_flow]

    async def input_fn(self, request: "Request") -> TParsedInputType:
        """Default input parser that extracts JSON body from the request."""
        content_type = request.headers.get("Content-Type", "").split(";")[0].strip()
        handler = self._DEFAULT_INPUT_HANDLERS.get(content_type)
        if handler is None:
            raise ValueError(f"Unsupported Content-Type: {content_type}")
        return await handler(request)

    def preprocess_fn(self, parsed_inputs: TParsedInputType) -> t.Dict[str, t.Any]:
        """Default converter that assumes parsed inputs are already in the correct format for graph execution."""
        if isinstance(parsed_inputs, list):
            return {"data": parsed_inputs}
        
        ret = pl.LazyFrame(parsed_inputs["data"])
        if "parameters" in parsed_inputs:
            ret = ret.with_columns(pl.lit(parsed_inputs["parameters"]).alias("parameters"))
        return parsed_inputs

    def output_fn(self, preprocessed_inputs: pl.LazyFrame, result: pl.LazyFrame) -> "pl.DataFrame":
        return pl.LazyFrame(result).collect()
    
    def format_response_fn(self, request: "Request", result: pl.DataFrame) -> "Response":
        accept_types = request.headers.get("Accept", "").split(",")
        for accept_type in accept_types:
            accept_type = accept_type.split(";")[0].strip()
            formatter = self._DEFAULT_RESULT_FORMATTERS.get(accept_type)
            if formatter: return formatter(result)
        raise ValueError(f"Unsupported Accept header: {request.headers.get('Accept')}")


    async def request_fn(self, request: "Request") -> "Response":
        """Default request handler that parses inputs, selects a flow, executes it, and returns the response."""
        parsed_inputs = await self.input_fn(request)
        # Do this earlier as a way to do early validation of the inputs.
        preprocessed_inputs = self.preprocess_fn(parsed_inputs)
        flow_config = self.select_flow_fn(parsed_inputs)
        graph = await flow_config.get_graph(parsed_inputs)
        result = graph.execute(preprocessed_inputs)
        output = self.output_fn(preprocessed_inputs, result)
        return self.format_response_fn(request, output)

    async def handle(self, request: "Request") -> "Response":
        return await self.request_fn(request)
    
    async def cleanup_fn(self):
        """Optional cleanup logic to run on server shutdown."""
        pass
    
# Dynamically find all methods ending with _fn or load_flows
RequestHandler.OVERRIDABLE_METHODS = [
    name for name in dir(RequestHandler) 
    if (name.endswith('_fn')) 
    and not name.startswith('default_') 
    and callable(getattr(RequestHandler, name, None))
]