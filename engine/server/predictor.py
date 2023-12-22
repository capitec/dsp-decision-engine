# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import re
import sys
import typing
import logging
from types import ModuleType

from contextlib import asynccontextmanager

from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from starlette import status
from engine.serving import RuleServer, get_config_class_from_module


prefix = os.environ.get("MODEL_PREFIX", "/opt/ml/")
model_path = os.path.join(prefix, "model")
sys.path.insert(1, model_path)
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
logger = logging.getLogger(__name__)


def load_config_class():
    try:
        import config
    except ImportError:
        return None
    return get_config_class_from_module(config)


@lru_cache(1)
def get_model() -> RuleServer:
    rs = load_config_class()
    if rs is not None:
        return rs
    return RuleServer()

# model = get_model()
# model.configure_logger(logger)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model into cache
    model = get_model()
    model.configure_logger(logger)
    yield
    get_model.cache_clear()


# app = FastAPI(lifespan=lifespan)
app = FastAPI()

# from fastapi_cprofile.profiler import CProfileMiddleware
# app.add_middleware(CProfileMiddleware, enable=True, server_app = app, filename= "/Users/cp371651/Documents/Workspace/Upskilling/Projects/HamingtonHCL/progs/merchant_loan/out/profilingrun.pstats", strip_dirs=False, sort_by="cumulative")
# profiler_middleware = app.user_middleware[0]


@app.get("/ping")
async def ping(model: typing.Annotated[RuleServer, Depends(get_model)]):
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # TODO proper health check
    return JSONResponse("\n", status_code=(status.HTTP_200_OK if model.is_ready() else status.HTTP_404_NOT_FOUND))

# TODO see if there is a faster c library for parsing accepted types
def get_q_value(accept_type:str):
    accept_type = accept_type.split(";")
    if len(accept_type) == 1:
        return 1, accept_type[0].strip()
    accept_type, params, *_ = accept_type
    match = get_q_value.expr.match(params)
    if not match:
        q = 1
    else:
        q = float(match[1])
    return q, accept_type.strip()
get_q_value.expr = re.compile(r"q=(\d+\.?\d+)")


# Cache frequently requested types
@lru_cache(maxsize=128)
def parse_accepted_types(accept: str)->typing.List[str]:
    accept = [get_q_value(s.strip()) for s in accept.split(',')]
    sorted_accept = sorted(
        [q,i,v] for i,(q,v) in enumerate(accept)
    )
    return tuple([v for *_,v in sorted_accept])


@app.route("/invocations", methods=["POST"])
async def transformation(request: Request):
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    model: RuleServer = get_model()
    # Handling request directly in FastAPI looks worse then Flask.
    content_type = request.headers.get("content-type", None)
    try:
        model.validate_content_type(content_type)
    except ValueError as e:
        msg = str(e) or f"Unsupported media type {content_type}"
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=msg)
    
    accept_types = parse_accepted_types(request.headers.get("accept", "*/*"))
    try:
        model.validate_response_type(accept_types)
    except ValueError as e:
        msg = str(e) or f"Unsupported Accept Types {accept_types}"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg)
    
    data = await request.body()
    
    try:
        response = model.run(data, content_type, accept_types)
    except ValueError as e:
        msg = e.args[0] if len(e.args) > 0 else f"Invalid payload {data}"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg)

    return Response(
        content = response.content,
        status_code = response.status_code,
        headers =  response.headers,
        media_type = response.media_type,
    )
