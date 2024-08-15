import re
import typing
from functools import lru_cache

ACCEPTED_TYPES_LIMIT = 10

Q_VALUE_REX = re.compile(r"q=(\d+\.?\d+)")


def _get_q_value(accept_type: str):
    """HTTP provides a method to set priorities on accepted types"""
    accept_type = accept_type.split(";")
    if len(accept_type) == 1:
        return 1, accept_type[0].strip()
    accept_type, params, *_ = accept_type
    match = Q_VALUE_REX.match(params)
    if not match:
        q = 1
    else:
        q = float(match[1])
    return q, accept_type.strip()


# Cache frequently requested types
@lru_cache(maxsize=128)
def parse_accepted_types(accept: str) -> typing.Tuple[str]:
    accept = [_get_q_value(s.strip()) for s in accept.split(",")[:ACCEPTED_TYPES_LIMIT]]
    sorted_accept = sorted((q, i, v) for i, (q, v) in enumerate(accept))
    return tuple([v for *_, v in sorted_accept])
