# Initial version made use of FastAPI. Seems a bit bloated for the simple functionality
from functools import lru_cache
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route
from starlette.requests import Request
from starlette import status
from spockflow.inference.exceptions import APIException, UnsupportedEncoding, InvalidInputError
from spockflow.inference.settings import get_settings
from spockflow.inference.io import content_types

@lru_cache(maxsize=1)
def get_handler():
    from spockflow.inference.handler import ServingHandler
    return ServingHandler.from_model_project()


def extract_common_parameters(request: Request) -> dict:
    """
    Extracts common parameters from the request.
    :param request: The request.
    :return: A dictionary with the following keys:
        model_name: The name of the model.
        model_version: The version of the model.
        model_outputs: The outputs of the model.
        model_name_header: The name of the header containing the model name.
        model_version_header: The name of the header containing the model version.
    """
    settings = get_settings()
    params = dict()

    content_type = request.headers.get("content-type", None)
    if content_type is None:
        raise UnsupportedEncoding("Expected header \"content-type\". None provided.")
    params["content_type"] = content_type

    params["accept"] = request.headers.get("accept", "*/*")

    if settings.server_model_name_header is not None:
        params["model_name"] = request.headers.get(settings.server_model_name_header)

    if settings.server_model_version_header is not None:
       params["model_version"] = request.headers.get(settings.server_model_version_header)

    if settings.server_model_output_override_header is not None:
        model_outputs = request.headers.get(settings.server_model_output_override_header)
        if model_outputs is not None:
            params["output_overrides"] = model_outputs.split(",")

    return params


def ping(request: Request):
    return Response(status_code=status.HTTP_200_OK)


async def predict(request: Request) -> Response:
    handler = get_handler()
    
    return handler.transform_fn(
        input_data = await request.body(), 
        **extract_common_parameters(request),
    ).to_api()


async def predict_with_config(request: Request):
    from pydantic import ValidationError
    from .models import PredictWithConfigRequestModel
    params = extract_common_parameters(request)

    if params.get("model_version") is not None:
        raise InvalidInputError("Model version cannot be specified for this endpoint.")

    content_type = params["content_type"]
    if content_type != content_types.JSON:
        raise UnsupportedEncoding.from_content_type(content_type, [content_types.JSON])
    
    try:
        data = PredictWithConfigRequestModel(**(await request.json()))
    except ValidationError as e:
        raise InvalidInputError(str(e))
    
    handler = get_handler()

    return handler.transform_fn(
        input_data = data.payload, 
        **extract_common_parameters(request),
        content_type=data.content_type,
        config_override=data.config
    ).to_api()


async def visualize(request: Request):
    handler = get_handler()
    params = extract_common_parameters(request)
    subset_params = {k:v for k,v in params if k in ["model_name", "model_version"]}
    model = handler.model_cache.get(**subset_params)
    print(model.graph.nodes)
    # model.graph.display_all
    # TODO
    return Response(status_code=status.HTTP_200_OK)


def startup():
    if get_settings().server_init_on_startup:
        get_handler()


def shutdown():
    if get_handler.cache_info().currsize > 0:
        get_handler().shutdown_fn()


routes = [
    Route('/ping', ping, methods=['GET']),
    Route('/invocations', predict, methods=['POST']),
]

if get_settings().server_inference_with_config_endpoint:
    routes.append(
        Route('/invocations-with-conf', predict_with_config, methods=['POST'])
    )

if get_settings().server_include_visualize:
    routes.append(
        Route('/visualize', visualize, methods=['GET'])
    )

exception_handlers = {
    APIException: APIException.handle
}

app = Starlette(
    routes=routes, 
    on_startup=[startup],
    on_shutdown=[],
    exception_handlers=exception_handlers
)