"""`Text`: a string a compiled step reads in place, as its UTF-8 bytes."""
from __future__ import annotations

__all__ = ["Text"]


def _utf8(value: object) -> bytes | None:
    if isinstance(value, Text):
        return value.bytes
    return value.encode() if isinstance(value, str) else None


class Text:
    """A string column a compiled step reads without copying it.

    Annotate an input `Text` instead of `str` and the step stays in the kernel:
    `==`, `!=`, `len`, `in`, `startswith` and `endswith` compile and answer as
    they would on a `str`. Compare against a literal, a module-level constant or
    a `str` `param()`. Anything else a `str` can do (`.upper()`, slicing,
    f-strings) needs a `str` input, which runs in Python.

    A null is not a string: it equals nothing, not even another null, and holds
    no prefix, suffix or part. Declare it `Text | None` to accept one. An output
    declared `Text` is written as a `str` column, in Python.

    Example::

        from decikit.ops import Text

        def is_government(employer: Text) -> bool:
            return employer.startswith("GOV") or "DEPT" in employer

        Text("private") == "private"        # True
        Text(None) == Text(None)            # False
    """

    __slots__ = ("bytes", "text")

    def __init__(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            # The column is read as its UTF-8 bytes, the same refusal a kernel gives.
            raise TypeError(f"{type(value).__name__} is not a string")
        self.text = value
        self.bytes = None if value is None else value.encode()

    def __repr__(self) -> str:
        return f"Text({self.text!r})"

    def __eq__(self, other: object) -> bool:
        other = _utf8(other)
        return self.bytes is not None and other is not None and self.bytes == other

    def __len__(self) -> int:
        # Code points, as CPython counts them. A null has no length a kernel can
        # report through `len`, which may not be negative, so it reads as empty.
        return 0 if self.text is None else len(self.text)

    def __getitem__(self, k: int) -> int:
        # The kernel's `(address, byte length)`; only the length means anything here.
        return (0, -1 if self.bytes is None else len(self.bytes))[k]

    def startswith(self, prefix: object) -> bool:
        prefix = _utf8(prefix)
        return self.bytes is not None and prefix is not None and self.bytes.startswith(prefix)

    def endswith(self, suffix: object) -> bool:
        suffix = _utf8(suffix)
        return self.bytes is not None and suffix is not None and self.bytes.endswith(suffix)

    def __contains__(self, part: object) -> bool:
        part = _utf8(part)
        return self.bytes is not None and part is not None and part in self.bytes
