from dataclasses import dataclass
from contextlib import contextmanager
from warnings import warn


@dataclass
class PEDErrorResponse:
    message: str
    details: str | None = None
    # Do we include a stack trace here? Maybe only in debug mode?

class PEDError(Exception):
    """Base class for all PED-related exceptions."""
    _STATUS_CODE = 500  # Default to Internal Server Error, can be overridden by subclasses
    _MESSAGE = "An error occurred in the PED system."

    def __init__(self, message=None, *args):
        self.message = message or self._MESSAGE
        super().__init__(self.message, *args)
    
    def get_status_code(self) -> int:
        """Return the HTTP status code associated with this error."""
        return self._STATUS_CODE
    
    def get_response_body(self) -> PEDErrorResponse:
        """Return the response body to be sent to the client."""
        return PEDErrorResponse(
            message=self.message,
            details=str(self)  # Include the exception message as details
        )


class PEDMissingDependencyError(PEDError, ModuleNotFoundError):
    """Raised when there is an error importing a module or source."""
    _STATUS_CODE = 500
    _MESSAGE = "Failed to import a required module or source. Please ensure the optional package is installed and available."
    # TODO make this more dynamic by reading from the pyproject.toml 
    _KNOWN_MODULES = {
        "pandera": "pandera>=0.29.0<1.0.0",
    }

    def __init__(self, package_name: str = None, optional_source: str = None, *args):
        self.optional_source = optional_source

        package_name = package_name or 'a package'
        if self.optional_source:
            self.message = (
                f"Failed to import {package_name} provided in {self.optional_source}. "
                f"Please ensure you install ped with pip install ped[{self.optional_source}] "
                f"or install {package_name} directly with pip install '{self._KNOWN_MODULES.get(package_name, package_name)}'."
            )
        else:
            self.message = (
                f"Failed to import {package_name}. "
                f"Please ensure you install {package_name} directly with pip install '{self._KNOWN_MODULES.get(package_name, package_name)}'."
            )
        super().__init__(self.message, *args)


@contextmanager
def wrap_import_errors(optional_source: str = None, raise_error=True):
    try:
        yield 
    # We only care about module not found error not import errors
    # as if we importing pandora it will raise module not found if the dependency is missing
    except ModuleNotFoundError as e:
        error = PEDMissingDependencyError(
            optional_source=optional_source, 
            package_name=e.name,
        )
        if raise_error:
            raise error from e
        else:
            warn(error.message + " Some functionality may not work properly.", ImportWarning)