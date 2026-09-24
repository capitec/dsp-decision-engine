"""Inference handler for credit limit management pipeline."""
from datetime import date
from pipeline import build
from decider import RequestHandler


class CreditLimitHandler(RequestHandler):
    """Request handler for credit limit management."""

    def preprocess(self, request):
        """Preprocess incoming request."""
        # Parse decision_date if provided as string
        if 'decision_date' in request and isinstance(request['decision_date'], str):
            request['decision_date'] = date.fromisoformat(request['decision_date'])
        return request

    def postprocess(self, response):
        """Postprocess response."""
        # Convert date objects to ISO strings for JSON serialization
        for key, value in response.items():
            if isinstance(value, date):
                response[key] = value.isoformat()
        return response


def handler(event, context):
    """Lambda-style handler for deployment."""
    h = CreditLimitHandler(build())
    return h.handle(event, context)
