# Not sure if there is a better way than this. Code from
# https://stackoverflow.com/questions/77255184/inheriting-all-init-arguments-type-hints-from-parent-class
import typing as t
from functools import update_wrapper

P = t.ParamSpec("P")
T = t.TypeVar("T")

def inherit_signature_from(
    original: t.Callable[P, T]
) -> t.Callable[[t.Callable], t.Callable[P, T]]:
    """Set the signature of one function to the signature of another."""
    def wrapper(f: t.Callable) -> t.Callable[P, T]:
        return update_wrapper(f, original)
    return wrapper
