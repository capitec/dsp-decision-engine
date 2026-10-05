from decider.exceptions import DeciderError


class UnsupportedContentTypeError(DeciderError):
    _STATUS_CODE = 415
    _MESSAGE = "Unsupported content type."


class InputParsingError(DeciderError):
    _STATUS_CODE = 400
    _MESSAGE = "Failed to parse request input."


class UnsupportedAcceptError(DeciderError):
    _STATUS_CODE = 406
    _MESSAGE = "Unsupported Accept media type."


class OutputFormattingError(DeciderError):
    _STATUS_CODE = 500
    _MESSAGE = "Failed to format the output."
