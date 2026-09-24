"""
Request handler for serving the business credit assessment pipeline.
"""
from decider import RequestHandler
from pipeline import build


handler = RequestHandler(build())
