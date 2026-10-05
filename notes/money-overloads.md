# Money operators on a cents type

**Verdict: GO.** `a + b`, `a * rate`, `a / n`, `a < b` on a `Money(cents=True)` value cost nothing
measurable on the serving path, give the same answer in every mode over an edge corpus, and fuse and
pack like any other step. The traps below are all fixable in the real build; none of them is a reason
to keep only the explicit `ops.decimal` functions.

The question came from `kernel-functions.md` ("Money overloads"). The prototype is throwaway; the
shape that worked is what the real build should copy.

## What was built

- **Python side:** `Cents(int)`, with `+ -` (Cents only), `*` and `/` by int and float, `/` by `Cents`
  (a float ratio), unary `-`, `abs`, and comparisons as for int. Everything else a plain `int` would answer
  (`Cents + 5`, `5 + Cents`, `Cents * Cents`, `//`, `%`, `**`) raises `TypeError`.
- **Kernel side:** `CentsType`, a numba type with an int64 model. Operators are registered on
  `CentsType` only (`+ - * /`, `+= -= *= /=`, unary `-`, `abs`, `min`, `max`, `int`, `float`, the six
  comparisons, and the constructor `Cents(n)`). No overload touches `int64`: `(1, 2) == (1, 2)`,
  `x * y` on plain ints and `len` are unchanged.
- **The edge:** the type exists only inside a step. A new source tag in the fused kernel builder wraps
  a column, a variable, an earlier result or a param as `CentsType` when the reader's annotation is
  `Money(cents=True)` (ten lines, `kernel.py`). Columns, `State`, params bundles and outputs stay plain
  int64; `lower_cast(CentsType, Integer)` turns a returned `Cents` back into the stored int64. The
  signature probe types a money input or param as `CentsType` (and `Optional(CentsType)` for
  `T | None`). The interpreted runner and the Python fallback wrap the same values in `Cents`.
- **Rounding mode in the type:** `Money(cents=True, rounding=...)`. `CentsType` carries the mode
  (`HALF_UP`, `HALF_EVEN`, `DOWN`, or none for a bare `Cents(n)`), and so does a `Cents` subclass in
  Python. `*` and `/` round to whole cents by the left operand's mode.
- **Rounding helper:** `round_whole(x, mode)` in `decider/ops/decimal.py`, one `@register_jitable`
  body. It works on the magnitude: `a - floor(a)` is exact for every double, and a tie is exactly
  `0.5`. Checked against `decimal.Decimal(x).to_integral_value(...)` for all three modes, on exact
  ties, `0.49999999999999994`, values around 2^52 and 2^53, and random products: 24,063 comparisons, 0
  mismatches, in Python and in a kernel. NaN and infinity raise `ValueError` in both.

## Numbers

Box idle but not quiet (load 11 on 28 cores); ratios are the result. Best of 5 repetitions of 20,000
`score()` calls (3,000 interpreted), after 2,000 warm-up calls. The flow is three steps (`net`, `fee`
with a money param `floor` and a float `rate`, `split`). "plain" is `int` cents with
`round_whole(x * rate, 0)` spelled out; "cents" is the same arithmetic as operators on `Cents`.
Outputs are identical.

| mode | plain p50 / p99 (µs) | cents p50 / p99 (µs) |
|---|---|---|
| fused | 32.6 / 44.8 | **32.4 / 40.3** |
| stepped | 56.0 / 79.1 | **55.7 / 71.4** |
| interpreted | 89.2 / 128.0 | 101.0 / 113.6 |

| batch, 200k rows | plain | cents |
|---|---|---|
| fused | 74.0M rows/s | 73.1M rows/s |
| stepped | 87.4M rows/s | 84.4M rows/s |

- **Compiled modes: no cost.** The wrap is a no-op bitcast and the operators inline to int64 and
  float64 arithmetic. This sits inside the per-request budget in `serving-latency.md` (30.4 µs
  flagship `score()`, 7 µs engine floor) with nothing added.
- **Interpreted: +12 µs per record** (+13%), from `Cents` method calls in Python. Not a serving mode.
- **First call, fresh process:** plain 0.75 s, cents 0.87 s with a cold step cache; 0.42 s and 0.40 s
  with it warm. A step signature containing `CentsType` round-trips through numba's disk cache. 8
  threads x 3,000 `score()` calls: 0 mismatches.
- **A `Cents` object must never cross the dispatcher.** Passed to an njit function it costs 3.0 µs
  against 0.23 µs for an int, and inside a tuple (the shape of a params bundle) **24 µs against
  0.4 µs**, because numba types it through the slow Python path on every call. This is why the type is
  made at the kernel edge from int64 and nothing else: bundles hold ints, and no dispatcher ever sees a
  `Cents`. Box and unbox are still registered, for `@helper` calls and tests.

## Typing in fused and packed control flow

- A money chain inside one fused kernel, a branch with money arms and a loop carrying a money balance
  (`balance * (1.0 + rate) - payment`, condition `balance > 0`) all fuse and pack: `packed` holds the
  branch and the loop, there are no fallback warnings, and `assert_equivalent` passes.
- Also pass: two money outputs from one step, `missing_as` on a money input, `T | None` money inputs,
  a money value read by a `float` step, `Cents(n)` from an int input, a `for` loop with `if` and `+=`
  inside a step, money params overridden by a params document and validated (`ge=0`; a float for a
  cents param is rejected by the existing validation).

## Equivalence

`decider.testing.assert_equivalent` (interpreted, stepped, fused, `score()`, session) over 253 rows
and 11 steps: ties at `.5` for positive and negative amounts (`5, 15, 25, -25, 1005, -1005 ...`) against
rates `0.5, -0.5, 0.1, 0.125, 0.0175, 1/3, 1.5, 2.5, -2`; amounts around 2^53; divisors `1, -1, 2, 3,
4, -7, 8`; all three rounding modes. A second run adds amounts up to 2^62 and the int64 bounds for the
comparison, ratio and float steps. All pass.

What does not match, by design: **overflow.** `Cents(2**62) + Cents(2**62)` is a big int in Python and
raises `OverflowError` when stored in the int64 column; a kernel wraps to -2^63 with no error. Likewise
`balance / n` for `2^63 - 1` goes through float64 and lands on 2^63. `assert_equivalent` reports it,
because the first mode raises. Overflow stays the checks' job (DCD101), as decided.

## `Cents + int` is a type error, both sides

- **Python:** `TypeError: unsupported operand for Cents + int: write Cents(n) to say an int is cents`.
- **Kernel:** a numba `TypingError` whose own line reads `cents[HALF_UP] + int64 is a type error: write
  Cents(n) to say an int is cents`. The step then falls back to Python with a `FallbackWarning` (and
  raises under `strict_compile=True`), where the Python `TypeError` above is what the user finally
  sees. So the spelling fails in every mode, but the kernel's message is 43 lines of numba candidate
  listing with the real line in the middle. The real build should cut a money `TypingError` down to
  that line in `_why`.

## Traps found

1. **The wrap must cover every source kind** (column, optional column, earlier result, loop/branch
   variable). The first cut wrapped columns only. A money loop then failed to type inside the packed
   kernel (`float64 - cents`), and `FusedRunner._packed` swallowed the error and dropped the packing
   for the life of the runner with **no warning**: answers stayed right, `packed` came back empty,
   nothing else showed it. The real tests must assert the path is in `packed` for a money loop and
   branch, and `_drop` should warn.
2. **`float op Cents` leaks in Python.** `1.5 + Cents(3)` and `1.5 * Cents(3)` return a plain float
   because `float.__add__` accepts any int subclass and never reaches `Cents.__radd__`. The kernel
   rejects both (falling back to Python, which then answers a float), so the answer agrees but is not
   a `Cents` and nothing warns. The only complete fix is for `Cents` not to subclass `int` (then
   `float.__add__` returns `NotImplemented` and `__radd__` can raise), at the cost of `isinstance(x,
   int)`, polars and numpy conversions. Decision for the real build; see Open.
3. **`NotImplemented` falls through to `int`.** `Cents + 5` returning `NotImplemented` still answers
   through `int.__radd__`, and `Cents // 2` through `int.__rfloordiv__`. Every operator that must not
   work has to raise explicitly.
4. **Python's `sum(xs)` starts at the int `0`**, so it raises on `Cents`; `sum(xs, Cents(0))` is the
   spelling, and works in a kernel after the mode unification below. Numba does not fall back from `+=`
   to `+` for an overloaded type, so `iadd`, `isub`, `imul`, `itruediv` are registered too. Numba's
   `min` and `max` do not accept a custom type; both are overloaded on `CentsType` pairs.
5. **The mode lives in the type, so modes have to unify.** A bare `Cents(n)` has no mode and takes the
   other side's (`CentsType.unify` and `can_convert_to`); two different concrete modes in `+`, `-`, or
   one returned value are a type error in the kernel and in Python. A ternary returning a `HALF_EVEN`
   and a `HALF_UP` value fails to unify in the kernel and falls back; Python would return either.
6. **`return 0` in a money step does not type.** An int literal does not unify with `CentsType`, so
   `return x if x > floor else 0` falls back with a warning. Allowing it would let `total = 0; total
   += c` compile in the kernel while Python raises (`0 + Cents`). Strict is right; the message should
   say `Cents(0)`.
7. **`T | None` money needs `Optional(CentsType)`.** Left as plain int64, the kernel accepted `balance +
   5` that Python rejects. The optional source is wrapped as an optional cents value.
8. **`Money.to_json` changes shape** when `rounding` is added to the dataclass (an enum object in
   `asdict`): `tests/steps/test_fields.py` and a debug-bridge test pin the old dict. Emit the name, and
   omit the default, to keep both.
9. **The first `round_whole` was wrong.** `x - floor(x)` is not exact for a negative `x` near zero:
   `-0.49999999999999994` gave `-1` instead of `0` under `HALF_UP`. Caught by the Decimal comparison,
   not by `assert_equivalent` (all modes agreed on the wrong value). Keep that comparison as a test.
10. **`@helper(signatures=)` has no money type.** `_numba_type` accepts `float`, `int`, `bool`; a helper
    taking a `Cents` needs `Cents` there.

## What the real build must include

Files:

- `decider/_engine/compile/money.py` (type): `CentsType` keyed by mode, int64 model, `typeof`, box and
  unbox, `lower_cast` to `Integer` and between modes, `unify` and `can_convert_to`, and the `_raw` and
  `_wrap` intrinsics.
- `decider/_engine/compile/semantics/money.py` (operators): the overloads listed above plus the
  `Cents(n)` constructor typing, each raising a one-line `TypingError` for `Cents` with a plain int or
  float, and for two concrete modes.
- `decider/ops/decimal.py`: `Rounding`, the rounding function (with `digits`, `round_whole` is the
  `digits=0` case), and the Python `Cents` class with explicit raises for every operator it must not
  answer. It needs to be public, since users write `Cents(0)`.
- `decider/fields.py`: `Money(rounding=)` (and `scale=`, see Open), `to_json` as in trap 8.
- `decider/_engine/compile/kernel.py`: the wrap source tag, including `Optional`.
- `decider/_engine/compile/units.py`: `Layout.spec` wraps scalar-node sources and params (not row
  nodes); `Fallback.run` wraps for the Python path.
- `decider/_engine/compile/njit.py`: `_input_type`, the param type in `_probe_signature`, the helper
  signature type, and `SALT` covering `money.py`, `semantics/` and `ops/` by walking (the conventions
  test `test_cache_salt_covers_every_semantics_and_ops_file` already expects this).
- `decider/_engine/run/runners/interpreted.py`: wrap inputs and params; compute the wrap once per node
  at bind, not per call as the prototype's cached lookup does.
- `decider/_engine/run/runners/fused.py`: warn when a packed kernel is dropped (trap 1).
- Checks: DCD102 points at `Money(cents=True)`; DCD103 at `decimal.round`.

Tests:

- The Decimal comparison for all three modes (trap 9).
- `assert_equivalent` over the corpus above, one money step per operator, plus branch and loop.
- `packed` contains a money branch and a money loop.
- `Cents + int`, `int + Cents`, `Cents * Cents`, mixed modes: a `TypeError` in Python and a message
  naming `Cents(n)` from `strict_compile=True`.
- `(1, 2) == (1, 2)` and plain int arithmetic unchanged once the overloads are registered.
- Cache: a changed overload changes `SALT`.

Not needed: a change to `packed.py`, to the trace code, or to `State`.

## Open

- **`int` subclass or not** (trap 2). Subclass: users and tools see ints, but `float + Cents` silently
  answers a float in Python. Not a subclass: complete type safety, more conversions to write.
- **`Money(scale=n)`.** The type would be keyed by `(scale, mode)`. `Cents * float` generalises;
  a scaled `*` between two scaled values needs a rule that `Cents` never did.
- **Comparisons with a float** (`balance < 1.5`) work in Python and are a type error in the kernel,
  because int64 against float64 differs from Python above 2^53. Likely a type error in Python too.
- **Mixed modes** are an error here; allowing the left mode would be one line each side.
- **Division by zero** raises `ZeroDivisionError` in both, for `Cents / 0` and `Cents / Cents`.
- **Overflow in Python.** `Cents` could raise when leaving int64 so the interpreted mode wraps nowhere
  the kernel would not, but the kernel still can't report it without per-operation checks.

Prototype: `.scratch/` in the spike worktree (`equiv.py`, `flow_equiv.py`, `params_equiv.py`,
`typing_cases.py`, `round_check.py`, `bench_money.py`, `box_bench.py`).
