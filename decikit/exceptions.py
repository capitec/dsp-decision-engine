from __future__ import annotations

from contextlib import contextmanager


__all__ = [
    "DecikitError", "DecikitMissingDependencyError", "EngineError", "ExprError", "FallbackWarning", "HiddenStateWarning",
    "IRError", "MissingInputError", "NotIntegralError", "ParamsError", "RegistryError", "ShadowedTableWarning",
    "StrictCompileError", "WiringError",
]


class DecikitError(Exception):
    """Base class of every error decikit raises; catch it to handle them all.

    `_STATUS_CODE` is the HTTP status a server answers with (500 unless a subclass says otherwise).

    Example::

        try:
            exe.run(df)
        except DecikitError as e:
            print(e)
    """

    _STATUS_CODE = 500
    _MESSAGE = "An error occurred in the Decikit system."

    def __init__(self, message=None, *args):
        self.message = message or self._MESSAGE
        super().__init__(self.message, *args)


class FallbackWarning(UserWarning):
    """A compiled-mode step runs outside the shared kernel; the message names the step and says why.

    Example::

        warnings.filterwarnings("ignore", category=FallbackWarning)   # or @allow_fallback per step
    """


class HiddenStateWarning(UserWarning):
    """A step reads a Python object that no config document, diff or kernel can see; the message names it and where.

    `@allow_fallback` does not silence it: the check is about what a release
    check can compare, not about speed.

    Example::

        warnings.filterwarnings("ignore", category=HiddenStateWarning)
    """


class ShadowedTableWarning(UserWarning):
    """A params document sets a table whose default rows come from a config document, so the document's rows are ignored.

    Emitted when a config version is staged (`decikit build`, `RequestHandler.stage()`) for a
    `param_table(..., seeded_from="<document>")`. Run-time overrides (experiments) stay silent.
    Three ways to switch it off: filter the category, set `DECIKIT_BUILD__WARN_SHADOWED_TABLES=false`,
    or declare the table with `allow_override=True`.

    Example::

        warnings.filterwarnings("ignore", category=ShadowedTableWarning)
    """


class DecikitMissingDependencyError(DecikitError, ModuleNotFoundError):
    """An optional dependency isn't installed; the message names the extra that provides it."""

    def __init__(self, package_name: str, optional_source: str):
        self.optional_source = optional_source
        super().__init__(
            f"Failed to import {package_name} provided in {optional_source}. "
            f"Please ensure you install decikit with pip install decikit[{optional_source}] "
            f"or install {package_name} directly."
        )


class WiringError(DecikitError, ValueError):
    """A pipeline can't be wired: its message names the step path and suggests a fix.

    Raised for a bad step name, duplicate paths, a likely typo, a read of a
    name a branch arm keeps to itself, a dag cycle, a bad emit or drop.

    Example::

        try:
            pipeline.run(df)
        except WiringError as e:
            print(e)   # names the step path and suggests a fix
    """


class IRError(DecikitError, TypeError):
    """A step, or the IR it builds, is malformed.

    Raised for a signature that can't be wired, a branch condition that isn't
    one call, a node with no origin, a shared param declared with two types.
    """


class ParamsError(DecikitError, ValueError):
    """A node's params are invalid for the params document in use."""


class MissingInputError(DecikitError, ValueError):
    """A required input (no `missing_as()`, no `| None`) is null or absent in the data.

    `input`, `path` and `null_count` say which input of which step, and on how many rows.
    """

    _STATUS_CODE = 400

    def __init__(self, input: str, path: str, null_count: int, n_rows: int, absent: bool = False,
                 fix: str | None = None):
        self.input, self.path, self.null_count = input, path, null_count
        where = f"step '{path}'" if path else "the pipeline"
        found = ("is not in the input frame or record" if absent
                 else f"has {null_count} null row(s) of {n_rows}")
        super().__init__(
            f"input '{input}' of {where} is required but column '{input}' {found}. "
            + (fix or f"Fix the data, or declare `{input}: T = missing_as(fill)` or `{input}: T | None`.")
        )


class NotIntegralError(DecikitError, ValueError):
    """A float that is not a whole number (7.5, NaN, infinity) reached a value declared `int`.

    A whole float (`7.0`) is accepted; nothing is truncated. The message names the step or input.

    Example::

        exe.run({"count": 7.5})     # NotIntegralError: input 'count' is declared int but got 7.5 ...
    """

    _STATUS_CODE = 400


class EngineError(DecikitError, ValueError):
    """An `Engine` can't run as asked: an input frame holding a column the pipeline itself produces,
    or a step reading a hidden object under `strict_compile`.

    Example::

        flow(ratio).run(df_with_a_ratio_column)   # EngineError: 'ratio' is produced by this pipeline ...
    """


class StrictCompileError(EngineError):
    """A step that would run in Python, refused by `strict_compile=True`.

    Example::

        try:
            FusedEngine(strict_compile=True).bind(pipeline).run(df)
        except StrictCompileError as e:   # 'p/label runs in Python, row by row: ...'
            ...
    """


class RegistryError(DecikitError, LookupError):
    """A config's `type` tag names no class registered under the registry it is loaded into.

    Example::

        ConfigurableStep.resolve("tre")   # RegistryError: 'tre' is not a registered ... Did you mean 'tree'?
    """


class ExprError(DecikitError, ValueError):
    """An expression is outside the admitted grammar; the message names the offending construct."""


@contextmanager
def wrap_import_errors(optional_source: str):
    # Only a missing module: an ImportError from inside an installed package is a real bug.
    try:
        yield
    except ModuleNotFoundError as e:
        raise DecikitMissingDependencyError(e.name, optional_source) from e
