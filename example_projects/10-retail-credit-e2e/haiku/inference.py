"""Request handler for serving retail credit decisions."""
from decider import RequestHandler
from pipeline import build


class RetailCreditHandler(RequestHandler):
    """Handler for retail credit end-to-end flow."""

    def get_pipeline(self):
        """Get the decision pipeline (entry point 1 by default)."""
        return build(entry_point_code=1)
