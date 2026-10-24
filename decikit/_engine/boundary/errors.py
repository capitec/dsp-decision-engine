from decikit.exceptions import DecikitError


class ArrowKindError(DecikitError, TypeError):
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


class ArrowImportError(DecikitError, RuntimeError):
    """nanoarrow refused what `__arrow_c_stream__()` handed over."""
