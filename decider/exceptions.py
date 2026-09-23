from __future__ import annotations

from contextlib import contextmanager


class DeciderError(Exception):
    """Base class of every error decider raises; catch it to handle them all.

    `_STATUS_CODE` is the HTTP status a server answers with (500 unless a subclass says otherwise).

    Example::

        try:
            exe.run(df)
        except DeciderError as e:
            print(e)
    """

    _STATUS_CODE = 500
    _MESSAGE = "An error occurred in the Decider system."

    def __init__(self, message=None, *args):
        self.message = message or self._MESSAGE
        super().__init__(self.message, *args)


class DeciderMissingDependencyError(DeciderError, ModuleNotFoundError):
    """An optional dependency isn't installed; the message names the extra that provides it."""

    def __init__(self, package_name: str, optional_source: str):
        self.optional_source = optional_source
        super().__init__(
            f"Failed to import {package_name} provided in {optional_source}. "
            f"Please ensure you install decider with pip install decider[{optional_source}] "
            f"or install {package_name} directly."
        )


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


class WiringError(DeciderError, ValueError):
    """A pipeline can't be wired: its message names the step path and suggests a fix.

    Raised for a bad step name, duplicate paths, a likely typo, a read of a
    name a branch arm keeps to itself, a dag cycle, a bad emit or drop.

    Example::

        try:
            pipeline.run(df)
        except WiringError as e:
            print(e)   # names the step path and suggests a fix
    """


class IRError(DeciderError, TypeError):
    """A step, or the IR it builds, is malformed.

    Raised for a signature that can't be wired, a branch condition that isn't
    one call, a node with no origin, a shared param declared with two types.
    """


class ParamsError(DeciderError, ValueError):
    """A node's params are invalid for the params document in use."""


class MissingInputError(DeciderError, ValueError):
    """A required input (no `missing_as()`, no `| None`) is null or absent in the data.

    `input`, `path` and `null_count` say which input of which step, and on how many rows.
    """

    _STATUS_CODE = 400

    def __init__(self, input: str, path: str, null_count: int, n_rows: int, absent: bool = False):
        self.input, self.path, self.null_count = input, path, null_count
        where = f"step '{path}'" if path else "the pipeline"
        found = ("is not in the input frame or record" if absent
                 else f"has {null_count} null row(s) of {n_rows}")
        super().__init__(
            f"input '{input}' of {where} is required but column '{input}' {found}. "
            f"Fix the data, or declare `{input}: T = missing_as(fill)` or `{input}: T | None`."
        )


class ArrowKindError(DeciderError, TypeError):
    """A column's Arrow type can't be read as its declared kind.

    Carries `column`, `kind` and `arrow_type` when the import (rather than the
    dtype table) refused it.
    """


class NeedsKernelSplit(ArrowKindError):
    """A declared column has no flat-array form (List, Struct, Array, Object, Binary, an overflowing Decimal).

    The caller splits the kernel around the column; `column` and `dtype` name it.
    """

    def __init__(self, name: str, dtype: object, reason: str = ""):
        self.column = name
        self.dtype = dtype
        message = f"column '{name}' ({dtype}) has no flat-array extraction; needs the kernel-split escape"
        super().__init__(f"{message} ({reason})" if reason else message)


class ArrowImportError(DeciderError, RuntimeError):
    """nanoarrow refused what `__arrow_c_stream__()` handed over."""


class EngineError(DeciderError, ValueError):
    """An `Engine` can't run as asked: an unknown mode or validation setting, or an input frame
    holding a column the pipeline itself produces.

    Example::

        Engine().bind(pipeline, mode="jit")   # EngineError: unknown mode 'jit'; expected one of [...]
    """


class RegistryError(DeciderError, LookupError):
    """A config's `type` tag names no class registered under the registry it is loaded into.

    Example::

        ConfigurableStep.resolve("tre")   # RegistryError: 'tre' is not a registered ... Did you mean 'tree'?
    """


class ExprError(DeciderError, ValueError):
    """An expression is outside the admitted grammar; the message names the offending construct."""


@contextmanager
def wrap_import_errors(optional_source: str):
    # Only a missing module: an ImportError from inside an installed package is a real bug.
    try:
        yield
    except ModuleNotFoundError as e:
        raise DeciderMissingDependencyError(e.name, optional_source) from e
