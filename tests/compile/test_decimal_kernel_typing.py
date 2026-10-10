import operator
import random
import threading

import numpy as np
import pytest
from numba import njit
from numba.core.errors import TypingError

import decikit._engine.compile.njit  # noqa: F401  (registers the decimal type and overloads)
from decikit._engine.compile.decimal_type import DecimalType
from decikit._engine.compile.semantics.decimals import _divlu, _mul64, raw, wrapper
from decikit.ops import decimal
from decikit.ops.decimal import Currency, Decimal, Rounding

INT64_MAX = 2 ** 63 - 1


# ---- typing in control flow

def test_total_starting_at_int_zero_types_as_a_decimal_in_a_loop():
    w = wrapper(4)

    @njit
    def f(xs):
        total = 0
        for x in xs:
            total += w(x)
        return raw(total)

    xs = np.array([10000, 5, 12345], np.int64)
    assert f(xs) == 22350
    assert sum((Decimal.from_raw(int(x), 4) for x in xs), Decimal(0)).raw == 22350


def test_sum_with_a_decimal_start_and_the_constructor_in_a_kernel():
    w = wrapper(2)

    @njit
    def f(xs):
        total = Decimal(0)
        for x in xs:
            total = total + w(x)
        return raw(total + Decimal(3, 4)), Decimal(2, 2).raw, raw(Decimal(5))

    assert f(np.array([150, 250], np.int64)) == (4 * 10 ** 4 + 3 * 10 ** 4, 200, 5)


def test_branches_over_two_scales_take_the_larger_scale():
    w2, w4 = wrapper(2), wrapper(4)

    @njit
    def f(x, y, c):
        v = w2(x) if c else w4(y)
        return raw(v + 1)

    assert f(250, 25000, True) == 350 * 100 and f(250, 25000, False) == 35000


def test_return_zero_unifies_with_a_decimal():
    w = wrapper(2)

    @njit
    def f(x):
        a = w(x)
        return a if a > 0 else 0

    assert f(250) == Decimal.from_raw(250, 2) and f(-250) == 0
    assert isinstance(f(250), Decimal) and f(250).scale == 2 and f(-250).scale == 2


def test_optional_decimal_arithmetic_and_comparison():
    w = wrapper(2)

    @njit
    def f(x, flag):
        v = w(x) if flag else None
        if v is None:
            return -1
        return raw(v * 3 + 1)

    assert f(250, True) == 850 and f(250, False) == -1


def test_a_boolean_test_on_a_decimal_matches_python():
    w = wrapper(2)

    @njit
    def f(x):
        return 1 if w(x) else 0

    assert [f(0), f(1), f(-1)] == [0, 1, 1]


def test_raw_attribute_matches_python():
    w = wrapper(2)

    @njit
    def f(x):
        return (w(x) * 2).raw

    assert f(125) == (Decimal.from_raw(125, 2) * 2).raw == 250


# ---- boxing: a Decimal object crossing the dispatcher

def test_decimal_objects_box_and_unbox_with_their_scale():
    @njit
    def f(a, b):
        return a + b, a * b

    s, p = f(Decimal("2.50"), Decimal("0.0500"))
    assert (s.raw, s.scale, p.raw, p.scale) == (25500, 4, 1250, 4) and type(s) is Decimal
    s, _ = f(Currency(2, 6500), Currency(1))
    assert s == Decimal("3.65") and s.scale == 4


def test_a_currency_and_a_decimal_of_the_same_scale_share_one_compiled_signature():
    @njit
    def f(a):
        return a + 1

    f(Currency(1))
    f(Decimal("1.0000"))
    assert len(f.signatures) == 1 and isinstance(f.signatures[0][0], DecimalType) and f.signatures[0][0].scale == 4


def test_the_numba_type_is_keyed_by_scale_only():
    assert DecimalType(2) is DecimalType(2) and DecimalType(2) is not DecimalType(4)
    assert DecimalType(2).key == 2


def test_units_never_reach_the_kernel_type():
    @njit
    def f(a):
        return a

    f(Currency(1, units="R"))
    f(Currency(1, units="$"))
    assert len(f.signatures) == 1


# ---- errors and traps

UNDEFINED = {
    "a + 1.5": lambda a: a + 1.5, "1.5 + a": lambda a: 1.5 + a, "a - 1.5": lambda a: a - 1.5, "a * 'x'": lambda a: a * "x",
    "2 / a": lambda a: 2 / a, "a / 'x'": lambda a: a / "x",
}


@pytest.mark.parametrize("source", list(UNDEFINED))
def test_undefined_operand_mixes_are_typing_errors_naming_the_decimal(source):
    inner, w = njit(UNDEFINED[source]), wrapper(2)

    @njit
    def f(x):
        return inner(w(x))

    with pytest.raises(TypingError, match="Decimal"):
        f(1)


def test_a_float_added_to_a_decimal_says_a_float_is_not_allowed():
    w = wrapper(2)

    @njit
    def f(x):
        return w(x) + 1.5

    with pytest.raises(TypingError, match="a float is not allowed"):
        f(1)


@pytest.mark.parametrize("op,symbol", [(operator.floordiv, "//"), (operator.mod, "%"), (operator.pow, "**")])
def test_floordiv_mod_and_pow_name_the_operator(op, symbol):
    w = wrapper(2)

    @njit
    def f(x):
        a = w(x)
        return op(a, 2)

    with pytest.raises(TypingError, match=rf"decimal\[2\] {symbol.replace('*', r'[*]')} .* is not defined for a Decimal"):
        f(1)


def test_float_over_decimal_in_a_kernel_says_to_use_float():
    w = wrapper(2)

    @njit
    def f(x):
        return 1.5 / w(x)

    with pytest.raises(TypingError, match=r"float\(d\)"):
        f(1)


def test_a_float_multiplier_stays_a_decimal_not_a_float_in_both_modes():
    w = wrapper(4)

    @njit
    def f(x):
        return 1.5 * w(x)

    r = f(30000)
    assert type(r) is Decimal and r.raw == 45000 and (1.5 * Currency(3)).raw == 45000


def test_the_constructor_refuses_floats_and_non_literal_scales():
    @njit
    def from_float(x):
        return Decimal(x, 2)

    @njit
    def from_variable(n, s):
        return Decimal(n, s)

    with pytest.raises(TypingError, match="takes an int"):
        from_float(2.5)
    with pytest.raises(TypingError, match="literal"):
        from_variable(2, 2)


def test_plain_ints_and_floats_are_untouched_by_the_overloads():
    @njit
    def f(a, b, x, y):
        return (a, b) == (a, b), len((a, b)), a * b, a + b, a - b, a / b, a // b, a % b, a ** 3, a < b, -a, abs(a), max(a, b), min(a, b), x * y, x / y, round(x, 1), x < y

    args = (7, 3, 2.675, 0.5)
    assert f(*args) == ((7, 3) == (7, 3), 2, 21, 10, 4, 7 / 3, 2, 1, 343, False, -7, 7, 7, 3, 2.675 * 0.5, 2.675 / 0.5, round(2.675, 1), False)


def test_uint64_and_int32_operands_follow_the_integer_rules():
    w = wrapper(2)

    @njit
    def f(x, i):
        return raw(w(x) + i), raw(w(x) * i)

    assert f(250, np.int32(3)) == (550, 750) and f(250, np.uint8(3)) == (550, 750)


# ---- 128-bit arithmetic

@njit
def mul64(a, b):
    return _mul64(a, b)


@njit
def divlu(u1, u0, v):
    return _divlu(u1, u0, v)


rng = random.Random(11)


def u64s(n: int) -> list[int]:
    edge = [0, 1, 2, 3, 10, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1, 2 ** 63 - 1, 2 ** 63, 2 ** 63 + 1, 2 ** 64 - 2, 2 ** 64 - 1, 10 ** 18, 10 ** 19]
    return edge + [rng.getrandbits(rng.randint(1, 64)) for _ in range(n)]


def test_mul64_gives_the_exact_128_bit_product():
    for a in u64s(3000):
        for b in u64s(0):
            hi, lo = mul64(np.uint64(a), np.uint64(b))
            assert (int(hi) << 64) + int(lo) == a * b, (a, b)
    for a, b in zip(u64s(20000), reversed(u64s(20000))):
        hi, lo = mul64(np.uint64(a), np.uint64(b))
        assert (int(hi) << 64) + int(lo) == a * b


def test_divlu_matches_python_ints_over_edge_and_random_cases():
    cases = [(u1, u0, v) for v in u64s(300) if v for u1 in {0, 1, v - 1, v // 2, v // 3} if u1 < v for u0 in (0, 1, 2 ** 63, 2 ** 64 - 1, v)]
    cases += [(rng.randrange(v), rng.getrandbits(64), v) for v in (rng.getrandbits(rng.randint(1, 64)) or 1 for _ in range(20000))]
    for u1, u0, v in cases:
        q, r = divlu(np.uint64(u1), np.uint64(u0), np.uint64(v))
        assert (int(q), int(r)) == divmod((u1 << 64) + u0, v), (u1, u0, v)


def test_no_division_helper_aborts_the_process_with_a_missing_i128_symbol():
    wa, wb = wrapper(4), wrapper(2)

    @njit(nogil=True)
    def kernel(x, y):
        a, b = wa(x), wb(y)
        return raw(a * b), raw(a / b), raw(b / a), raw(decimal.round(a, 1, Rounding.HALF_EVEN)), a < b

    assert kernel(INT64_MAX // 3, 12345) is not None
    signature = kernel.signatures[0]
    text = kernel.inspect_llvm(signature) + kernel.inspect_asm(signature)
    for banned in ("udivti3", "divti3", "umodti3", "modti3", "udiv i128", "sdiv i128", "urem i128", "srem i128"):
        assert banned not in text, banned


def test_kernels_run_concurrently_without_the_gil():
    wa, wb = wrapper(4), wrapper(2)

    @njit(nogil=True)
    def many(x, y):
        a, b = wa(x), wb(y)
        return raw(a + b), raw(a * b), raw(a / b) if b.raw else 0, a < b

    corpus = [(a, b) for a in (0, 1, -7, 12345, 2 ** 40, INT64_MAX // 5) for b in (1, -3, 99, 2 ** 33)]
    want = [many(a, b) for a, b in corpus]
    failures = []

    def work():
        for _ in range(3):
            if [many(a, b) for a, b in corpus] != want:
                failures.append(1)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures


def test_the_cache_salt_reads_the_decimal_files():
    import hashlib
    from pathlib import Path

    from decikit._engine.compile import semantics

    for name in ("decimal_type.py", "semantics/decimals.py"):
        assert any(p.endswith(name) for p in map(str, semantics._FILES)), name
    assert any(str(p).endswith("ops/decimal.py") for p in semantics._FILES)
    assert semantics.SALT == hashlib.sha256(b"".join(Path(p).read_bytes() for p in semantics._FILES)).hexdigest()


_CACHE_CHILD = """
import json
import decikit._engine.compile.njit
from numba import njit
from decikit._engine.compile.semantics.decimals import raw, wrapper
from decikit.ops.decimal import Decimal
w = wrapper(4)

@njit(cache=True, nogil=True)
def f(x, y):
    return raw(w(x) * 3 + w(y) / 7)

@njit(cache=True)
def g(a, b):
    return a * b - 1

out = [f(10000, 25), g(Decimal("2.5"), Decimal("0.05")).raw]
print(json.dumps([out, len(f.stats.cache_hits), len(f.stats.cache_misses), len(g.stats.cache_hits), len(g.stats.cache_misses)]))
"""


def test_a_kernel_with_decimal_types_is_reused_from_the_disk_cache_by_a_second_process(tmp_path):
    import json
    import os
    import subprocess
    import sys

    (tmp_path / "child.py").write_text(_CACHE_CHILD)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"))

    def run():
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    assert run() == [[30004, -88], 0, 1, 0, 1]
    assert run() == [[30004, -88], 1, 0, 1, 0]
