import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    import polars as pl
    from starlette.requests import Request
    from starlette.responses import Response
    from ped.flow import FlowConfiguration


handler: "RequestHandler" = None

async def initialize_request_handler():
    global handler

    import sys
    from pathlib import Path
    from ped.settings import settings
    from ped import initialize_ped


    root_path = Path(settings.api.root_path)
    sys.path.insert(0, str(root_path.resolve()))
    # Must maybe think about this a bit however i think the ped_extensions should be relative to the api root path
    settings.ext.extension_path = str((root_path / settings.ext.extension_path).resolve())
    # Load all extensions
    initialize_ped()

    handler = RequestHandler()

    # The below can import from this module and update the handler with custom logic
    if settings.api.init_module:
        init_file = root_path / (settings.api.init_module+".py")
        if init_file.exists():
            from importlib import import_module
            import_module(settings.api.init_module)

    await handler.initialize()


TParsedInputType = t.TypeVar("TParsedInputType")


async def default_load_flows(handler: "RequestHandler") -> t.Dict[str, "FlowConfiguration"]:
    """Load flow configurations from the configured extension path and return a mapping of flow name to configuration."""
    from ped.settings import settings
    from pathlib import Path
    from ped.flow import FlowConfiguration

    flow_path = Path(settings.api.root_path) / settings.api.flow_subpath
    flow_configs = {}
    for flow_file in flow_path.glob("*.json"):
        flow_name = flow_file.stem
        flow_configs[flow_name] = FlowConfiguration.model_validate(flow_file)

    return flow_configs



async def default_select_flow(handler: "RequestHandler", request: TParsedInputType) -> "FlowConfiguration":
    """Select a flow configuration based on the incoming request. By default, this returns the flow named 'default'."""
    if handler.flow_catalog is None:
        raise ValueError("Flow catalog not loaded")

    if handler.default_flow not in handler.flow_catalog:
        raise ValueError(f"Default flow '{handler.default_flow}' not found in flow catalog")

    return handler.flow_catalog[handler.default_flow]


# TODO Flesh this out to make it work for any type of input
async def default_parse_inputs(request: "Request") -> TParsedInputType:
    """Default input parser that extracts JSON body from the request."""
    return request.json()

async def default_preprocess_inputs(parsed_inputs: TParsedInputType) -> t.Dict[str, t.Any]:
    """Default converter that assumes parsed inputs are already in the correct format for graph execution."""
    return parsed_inputs # TODO PL.LazyFrame

async def default_format_response(handler: "RequestHandler", result: pl.LazyFrame, response_format:str) -> "Response":
    """Default response formatter that converts the result to JSON."""
    if response_format == "application/json":
        return pl.LazyFrame(result).collect().to_dicts()
    else:
        raise ValueError(f"Unsupported response format: {response_format}")


def parse_response_format(request: "Request") -> str:
    """Parse the desired response format from the request headers or parameters. Defaults to 'application/json'."""
    # TODO some more checks like if its "*/*" maybe we reply with a default i also want to strip extra info. Maybe this should be in the default format response though.
    return request.headers.get("Accept", "application/json")

async def default_handle_request(handler: "RequestHandler", request: "Request") -> "Response":
    """Default request handler that parses inputs, selects a flow, executes it, and returns the response."""
    parsed_inputs = await handler.parse_inputs(handler, request)
    # Do this earlier as a way to do early validation of the inputs.
    preprocessed_inputs = await handler.preprocess_inputs(handler, parsed_inputs)
    flow_config = await handler.select_flow(handler, parsed_inputs)
    graph = await flow_config.get_graph(parsed_inputs)

    result = graph.execute(preprocessed_inputs)

    response_format = parse_response_format(request)
    return await handler.format_response(handler, result, response_format)


@dataclass
class RequestHandler(t.Generic[TParsedInputType]):
    flow_catalog: t.Dict[str, "FlowConfiguration"] = None
    default_flow: str = "default"

    load_flows: t.Callable[["RequestHandler"], t.Awaitable[t.Dict[str, "FlowConfiguration"]]] = default_load_flows

    handle_request: t.Callable[["RequestHandler", "Request"], t.Awaitable["Response"]] = None  # To be set by the user of the handler, e.g. in the API module
    parse_inputs: t.Callable[["RequestHandler", "Request"], t.Awaitable[TParsedInputType]] = default_parse_inputs
    select_flow: t.Callable[["RequestHandler", TParsedInputType], t.Awaitable["FlowConfiguration"]] = default_select_flow
    preprocess_inputs: t.Callable[["RequestHandler", TParsedInputType], t.Awaitable[t.Dict[str, t.Any]]] = default_preprocess_inputs


    async def initialize(self):
        self.flow_catalog = await self.load_flows(self)

    async def handle(self, request: "Request") -> "Response":
        return await self.handle_request(self, request)