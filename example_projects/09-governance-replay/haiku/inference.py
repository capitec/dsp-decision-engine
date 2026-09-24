"""Handler for governance and replay harness.

Serves the harness over HTTP or as a Python function.
"""
import json
import sys
from datetime import date
from pipeline import build
from decider import Engine


def handle(request: dict) -> dict:
    """Handle a governance harness request.

    Args:
        request: Request with structure:
            {
                "operation": "replay" | "explain" | "diff" | "swap_set",
                "parameters": {...}
            }

    Returns:
        Response dict with results.
    """
    operation = request.get("operation", "replay")
    params = request.get("parameters", {})

    try:
        harness = build()
        engine = Engine(pipeline=harness)

        result = engine.run(
            step='route_to_capability',
            operation=operation,
            **params,
        )
        return {"status": "success", "result": result}

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "exception_type": type(e).__name__,
        }


def _parse_date(s):
    """Parse date string or return today."""
    if not s:
        return date.today()
    if isinstance(s, date):
        return s
    if isinstance(s, str):
        return date.fromisoformat(s)
    return date.today()


if __name__ == "__main__":
    # Read request from stdin or file argument
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            request = json.load(f)
    else:
        request = json.load(sys.stdin)

    response = handle(request)
    print(json.dumps(response, indent=2, default=str))
