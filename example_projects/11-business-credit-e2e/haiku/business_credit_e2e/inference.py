"""Inference handler for business credit pipeline."""
from decider.serving import RequestHandler
from pipeline import build
import json
from datetime import date


class BusinessCreditHandler(RequestHandler):
    """Request handler for business credit decisions."""

    def __init__(self):
        self.pipeline = build()

    def handle_request(self, request: dict) -> dict:
        """Route request to appropriate entry point."""
        entry_point = request.get("entry_point", "ep1_origination")

        # Extract parameters
        params = request.get("parameters", {})

        # Route to appropriate step
        if entry_point == "ep1_origination":
            step_fn = self.pipeline["ep1_origination"]
        elif entry_point == "ep3_annual_review":
            step_fn = self.pipeline["ep3_annual_review"]
        elif entry_point == "covenant_test":
            step_fn = self.pipeline["covenant_test_dscr"]
        else:
            return {"error": f"Unknown entry point: {entry_point}"}

        # Convert string dates to date objects if present
        for key in ["decision_date_override", "review_date", "test_date"]:
            if key in params and isinstance(params[key], str):
                params[key] = date.fromisoformat(params[key])

        try:
            result = step_fn(**params)
            return {
                "entry_point": entry_point,
                "decision": result,
            }
        except Exception as e:
            return {
                "entry_point": entry_point,
                "error": str(e),
            }


handler = BusinessCreditHandler()


def predict(request_json: str) -> str:
    """Lambda-like interface."""
    request = json.loads(request_json)
    result = handler.handle_request(request)
    return json.dumps(result)
