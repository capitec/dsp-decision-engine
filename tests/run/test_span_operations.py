"""A `Raw[bytes]` value answers string operations in the kernel, exactly as CPython does."""
import polars as pl
import pytest
from numba.core import types
from numba.core.errors import TypingError

from decider import Engine, Raw, flow, param
from decider.engine.compile.span import SPAN, _span_eq

COMPILED = ("stepped", "fused")
# ASCII, empty, 2-byte, 3-byte, 4-byte and a null.
VALUES = ["private", "", "privé", "ééé", "café bar", "subénd",
          "\U0001f600x", None]
FRAME = pl.DataFrame({"sector": VALUES}, schema={"sector": pl.String})
WANT = "privé"


def eq_literal(sector: Raw[bytes] | None) -> bool:
    return sector == "privé"


def ne_literal(sector: Raw[bytes] | None) -> bool:
    return sector != "privé"


def eq_param(sector: Raw[bytes] | None, want: str = param("privé")) -> bool:
    return sector == want


def eq_global(sector: Raw[bytes] | None) -> bool:
    return sector == WANT


def code_points(sector: Raw[bytes] | None) -> int:
    return len(sector)


def starts(sector: Raw[bytes] | None) -> bool:
    return sector.startswith("priv")


def starts_param(sector: Raw[bytes] | None, prefix: str = param("priv")) -> bool:
    return sector.startswith(prefix)


def ends(sector: Raw[bytes] | None) -> bool:
    return sector.endswith("énd")


def holds(sector: Raw[bytes] | None) -> bool:
    return "afé" in sector


def is_null(sector: Raw[bytes] | None) -> bool:
    return len(sector) < 0


def byte_length(sector: Raw[bytes] | None) -> int:
    return sector[1]


CPYTHON = {
    "eq_literal": [s == WANT for s in VALUES],
    "ne_literal": [s != WANT for s in VALUES],
    "eq_param": [s == WANT for s in VALUES],
    "eq_global": [s == WANT for s in VALUES],
    "code_points": [-1 if s is None else len(s) for s in VALUES],
    "starts": [s is not None and s.startswith("priv") for s in VALUES],
    "starts_param": [s is not None and s.startswith("priv") for s in VALUES],
    "ends": [s is not None and s.endswith("énd") for s in VALUES],
    "holds": [s is not None and "afé" in s for s in VALUES],
    "is_null": [s is None for s in VALUES],
    "byte_length": [-1 if s is None else len(s.encode()) for s in VALUES],
}


@pytest.mark.parametrize("fn", [eq_literal, ne_literal, eq_param, eq_global, code_points,
                                starts, starts_param, ends, holds, is_null, byte_length])
@pytest.mark.parametrize("mode", COMPILED)
def test_span_operations_answer_as_cpython_does(mode, fn):
    exe = Engine(strict_compile=True).bind(flow(fn, name="p"), mode=mode)
    assert exe.run(FRAME)[fn.__name__].to_list() == CPYTHON[fn.__name__]
    assert exe.fallbacks() == {}


def unrelated_pairs(low: int, high: int) -> int:
    # The span operations must not capture a plain pair of ints.
    same = (low, high) == (1, 2)
    return len((low, high)) + (10 if same else 0) + (100 if (low, high) < (1, 3) else 0)


@pytest.mark.parametrize("mode", COMPILED)
def test_a_plain_pair_of_ints_keeps_its_own_operations(mode):
    exe = Engine(strict_compile=True).bind(flow(unrelated_pairs, name="p"), mode=mode)
    out = exe.run(pl.DataFrame({"low": [1, 1], "high": [2, 9]}))
    assert out["unrelated_pairs"].to_list() == [112, 2]


def test_a_string_built_at_run_time_is_refused_not_answered_wrongly():
    # numba folds a str literal, even a sliced one; only a value it cannot see reaches this guard.
    with pytest.raises(TypingError, match="not against a string built at run time"):
        _span_eq(SPAN, types.unicode_type)


@pytest.mark.parametrize("mode", COMPILED)
def test_score_reads_one_record_through_the_same_operations(mode):
    exe = Engine(strict_compile=True).bind(flow(eq_literal, name="p"), mode=mode)
    assert exe.score({"sector": WANT})["eq_literal"] is True
    assert exe.score({"sector": "public"})["eq_literal"] is False


def other(sector: Raw[bytes] | None, alias: Raw[bytes] | None) -> bool:
    return sector == alias


@pytest.mark.parametrize("mode", COMPILED)
def test_two_spans_compare_as_strings(mode):
    exe = Engine(strict_compile=True).bind(flow(other, name="p"), mode=mode)
    frame = pl.DataFrame({"sector": ["private", "public", None], "alias": ["private", "priv\u00e9", None]})
    assert exe.run(frame)["other"].to_list() == [True, False, False]


def test_the_span_operations_need_a_compiled_mode():
    # `Raw[bytes]` hands an interpreted run the span itself, which is not a str.
    exe = Engine().bind(flow(eq_literal, name="p"), mode="interpreted")
    assert exe.run(FRAME)["eq_literal"].to_list() == [False] * len(VALUES)
