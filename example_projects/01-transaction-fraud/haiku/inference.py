"""Request handling for fraud interdiction pipeline."""
from decider.serving.handler import RequestHandler


class Handler(RequestHandler):
    """Custom handler for fraud interdiction requests.

    Default behavior: decode JSON request, route to pipeline, encode response.
    Override any method to customize.
    """
    pass
