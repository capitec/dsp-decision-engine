"""Request handling for the shared credit core library."""
from decider.serving.handler import RequestHandler


class Handler(RequestHandler):
    """Custom handler for core library requests.

    Default behavior: decode JSON request, route to pipeline, encode response.
    Override any method to customize.
    """
    pass
