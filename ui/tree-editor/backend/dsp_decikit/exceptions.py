"""Custom exceptions for DSP Decikit application."""

from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from starlette import status as http_status
from urllib.parse import quote_plus
from pydantic import ValidationError


class HandledException(HTTPException):
    """Base class for exceptions that provide custom handling logic.

    This allows exceptions to define their own response behavior while still
    being valid HTTPExceptions that can be caught by FastAPI's middleware.
    """

    def handle(self, request: Request):
        """Override this method to provide custom response handling."""
        # Default behavior - just raise as normal HTTPException
        raise self


class UnauthorizedException(HandledException):
    """Exception raised when authentication or authorization fails."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=http_status.HTTP_401_UNAUTHORIZED)

    def handle(self, request: Request):
        """Handle by redirecting to unauthorized page with error details."""
        from dsp_decikit.settings import settings

        encoded_detail = quote_plus(self.detail)
        response = RedirectResponse(
            url=f"/unauthorized?detail={encoded_detail}",
            status_code=http_status.HTTP_302_FOUND,
        )

        # Clean up any auth-related cookies to ensure fresh state
        # Clear main session cookie
        response.delete_cookie(
            **settings.auth.cookie.model_dump(mode="json", exclude={"key"}),
            key=settings.auth.cookie.key,
        )

        # Clear session status cookie
        response.delete_cookie(
            **settings.auth.status_cookie.model_dump(mode="json", exclude={"key"}),
            key=settings.auth.status_cookie.key,
        )

        # Clear OIDC login session cookie if OIDC is enabled
        if settings.auth.oidc.enabled:
            response.delete_cookie(
                **settings.auth.oidc.cookie.model_dump(mode="json", exclude={"key"}),
                key=settings.auth.oidc.cookie.key,
            )

        return response


def handle_handled_exceptions(request: Request, exc: HandledException):
    """Generic handler for all HandledException subclasses."""
    return exc.handle(request)


def handle_pydantic_validation_error(request: Request, exc: ValidationError):
    """
    Handle Pydantic ValidationError and return structured 400 response.

    Returns a JSON response with:
    - message: Human-readable error summary
    - detail: Detailed error message
    - validation_errors: List of validation errors with field paths and messages
    """
    import json

    errors = exc.errors()

    # Build structured validation errors
    validation_errors = []
    for error in errors:
        # Handle input field that might contain non-JSON-serializable objects
        input_value = error.get("input")
        if input_value is not None:
            try:
                # Test if the input is JSON serializable
                json.dumps(input_value)
            except (TypeError, ValueError):
                # If not serializable, convert to string representation
                input_value = str(input_value)

        validation_errors.append(
            {
                "loc": error.get(
                    "loc", []
                ),  # Field path (e.g., ["project_data", "nodes", 0, "id"])
                "msg": error.get("msg", ""),  # Error message
                "type": error.get(
                    "type", ""
                ),  # Error type (e.g., "value_error", "missing")
                "input": input_value,  # The invalid input value (if available)
            }
        )

    # Create summary message
    field_count = len(validation_errors)
    message = f"Validation failed: {field_count} error{'s' if field_count != 1 else ''}"

    return JSONResponse(
        status_code=http_status.HTTP_400_BAD_REQUEST,
        content={
            "message": message,
            "detail": "Project data validation failed. Please check the validation_errors field for details.",
            "validation_errors": validation_errors,
        },
    )


def handle_request_validation_error(request: Request, exc: RequestValidationError):
    """
    Handle FastAPI RequestValidationError and return structured 400 response.

    This handles validation errors from FastAPI's automatic request validation
    (before the endpoint handler runs).
    """
    import json

    errors = exc.errors()

    # Build structured validation errors
    validation_errors = []
    for error in errors:
        # Handle input field that might contain non-JSON-serializable objects
        input_value = error.get("input")
        if input_value is not None:
            try:
                # Test if the input is JSON serializable
                json.dumps(input_value)
            except (TypeError, ValueError):
                # If not serializable, convert to string representation
                input_value = str(input_value)

        validation_errors.append(
            {
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
                "input": input_value,
            }
        )

    # Create summary message
    field_count = len(validation_errors)
    message = f"Validation failed: {field_count} error{'s' if field_count != 1 else ''}"

    return JSONResponse(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": message,
            "detail": "Request validation failed. Please check the validation_errors field for details.",
            "validation_errors": validation_errors,
        },
    )


def register_handlers(app):
    """Register all custom exception handlers with the FastAPI app."""
    app.exception_handler(HandledException)(handle_handled_exceptions)
    app.exception_handler(ValidationError)(handle_pydantic_validation_error)
    app.exception_handler(RequestValidationError)(handle_request_validation_error)
