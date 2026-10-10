# Money

Read this before changing `Decimal`, `Currency`, rounding or overflow behaviour: it lists the rules the code holds and why.

## Money is a fixed-point `Decimal`, not a float or int cents

- `Decimal[p, s]` is an int64 `raw` count of `10**-s` units; `Currency` is scale 4 (to 922,337,203,685,477.5807).
- Why: floats drift; an `int`-subclass `Cents` leaked (`1.5 + Cents(3)` answered a float), so `Decimal` is a plain class.
- Not: a rate held as scaled int (`round(rate * 1e9)`): +5 cents of total interest on R250,000 over 60 months, where float64 matched a 60-digit `Decimal` on all 20,000 instalments tested. Money that is split must allocate by largest remainder (11 float64 parts of R1,207.99 summed to R1,208.02).
- See: [experiment](../experiments/money-types-and-rounding.md), `decider/ops/decimal.py`, [kernels](../../architecture/engine/kernels.md)

## A kernel types the int64 at its edge

- `DecimalType(scale)` wraps columns, params and results as int64 at the kernel boundary; the object never crosses the dispatcher.
- Why: a Python object costs 3.5 µs per call, 14-21 µs inside a params tuple, against 0.3 and 0.4 for an int.
- Every source kind must be wrapped (column, optional, earlier result, loop or branch variable): a packed kernel that fails to type is dropped silently, so assert `packed`.

## The numba type is keyed by scale; precision is 19 digits at most

- `Currency` and `Decimal[19, 4]` share one signature; `units` never reaches a kernel or cache key.
- Why: a type per `(p, s)` only makes `Currency + Decimal[18, 2]` fall back for a range nobody has.
- Not: Decimal128 in kernels; `p > 19` raises at build.

## 128-bit work is hand-rolled uint64 arithmetic

- `*`, `/` and compares use `_mul64` and `_divlu`; no i128 division is emitted (pinned by `tests/compile/test_decimal_kernel_typing.py`).
- Why: LLVM's i128 divide calls `__udivti3` and aborts the process, uncatchably. Cost 6 ns, 21 ns for wide operands.

## Operators round half-even once per operation, like polars

- `+ - * /` give the larger scale; an int is whole units; a float may multiply or divide and returns a Decimal. `d + 1.5`, `float / d`, `//`, `%`, `**` and `round(d, n)` are errors.
- Why: polars is half-even with scale `max(s1, s2)`, so frame tier and step agree on ties; an earlier half-up rule was reversed.
- See: [Types and money](../../guides/types-and-money.md)

## Explicit rounding is `decimal.round(x, digits, mode)`

- `Rounding` is `HALF_UP` (default), `HALF_EVEN` or `DOWN`; a Decimal keeps its scale; the mode is an int so one compiled step serves all.
- Storing into a coarser output scale rounds half-even again; compute at one scale when a single rounding matters.
- Not: `rescale`, designed and not built.

## `round(x, n)` on a float rounds the exact double

- A kernel's `round(2.675, 2)` is `2.67`, as CPython, via `round_float` (Dekker two-product), one `@register_jitable` body for Python and kernels.
- Why: plain numba rounds the scaled float and gives 2.68; each such tie is a cent on an instalment.
- Test against `decimal.Decimal.quantize` for all modes; a negative-near-zero bug was agreed on by every mode and caught only that way.

## Constructors: floats round from the exact value, strings from digits

- `Currency(2.675)` is 2.67, `Currency("2.675")` is 2.68, `Currency(major, minor)` is exact with `0 <= minor < 10**scale`.
- Why: polars casts floats through the shortest repr (2.68); decider keeps the float's CPython meaning.

## Units are a display string, no locale library

- `units="R"` shows as `R 4.50`; it is excluded from `==` and `hash`, and the left operand's units survive when both sides agree.
- Why: Babel is 31 MB and prints `R4,50`; a `ZAR + USD` error would exist only in Python modes, breaking same-answer-in-every-mode.

## Past int64 Python raises and a kernel wraps; DCD101 flags it

- `OverflowError` in Python (`_fit`), low 64 bits in a kernel; `assert_equivalent` reports it.
- Why: a kernel cannot check per operation, and wrapping in Python too would hide the bug.
- Corpora include `.xx5` ties and large principals. See `decider/lab/checks/_builtin/numeric/money.py`.

## Running totals that can grow accumulate in float64

- Exact to 2^53 (about R90 trillion in cents); DCD101's fix text says so.
- Why: a sum-of-squares over cents wraps at 2,667 rows; fixed-point compound interest at R100,000 gave -R90,971.75 in a kernel.

## A spelling a kernel rejects fails in every mode

- It is a numba `TypingError`, so the step falls back (or raises under `strict_compile=True`) and the Python class raises `TypeError`.
- Gap: the reported reason is numba's candidate listing with decider's line in the middle.
