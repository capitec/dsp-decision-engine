# Functions that run in kernels

Today, kernel functions live in loose modules: string ops on spans, CPython-equal number semantics, and soon decimals and money. Credit functions are coming too (`credit-functions.md`). This note gives them all a home and one contract.

## What exists (2026-10-05)

| Kind | Where | What the user writes | How it reaches a kernel |
|---|---|---|---|
| Number semantics | `engine/compile/cpython.py`: `round(float, int)` with CPython's exact-double rounding; `float ** int` through libm `pow` | `round(x, 2)`, `(1 + r) ** 12` (nothing imported) | a global numba `@overload` on a builtin, registered when the module is imported |
| String semantics | `engine/compile/span.py`: `==`, `!=`, `len`, `startswith`, `endswith`, `in` and `[]` on a span (`Raw[bytes]` / `bytes` inputs, tree string features), with the `SpanType` and `SpanArray` boxing in the same file | `sector == "private"`, `sector.startswith(prefix)` | `@overload` / `@overload_method` on decider's own `SpanType` (`notes/strings-span-ops.md`) |
| User helpers | `steps/helpers.py`: `@helper(signatures=...)` | their own function, called from a step | `_prepare_function` (`njit.py:233`) swaps the global for a shared dispatcher |
| Internal kernels | tree walker, table matcher, scorecard (`register_jitable`), the expression postfix evaluator, Arrow intrinsics | nothing | private to their step type or to the boundary |
| Expression-language functions | `steps/expr`: `min`, `max`, `abs`, defined twice (the Python `_eval` table and the postfix `_CALLS` opcodes) | `expr: "min(a, b) > 3"` in a tree or flow file | its own opcode table |
| **Planned:** money rounding | `money-rounding-and-overflow.md` specifies one rounding helper for every mode. **It was never built.** | | |
| **Planned:** decimals | `Decimal` can't enter a kernel. The boundary already casts a Decimal column to scaled `int64` cents (`boundary/dtypes.py:65-79`). | | |

## What's wrong with that

1. **There is no home and no stated contract.** Two semantics modules sit loose in `engine/compile/`, and decimals, money and credit would add more. Nothing says what such a module must guarantee. The span-ops note had to learn the hard way that overloading a bare `UniTuple(int64, 2)` hijacks every `(a, b) == (c, d)` in user code.
2. **The cache salt is a hand-kept list.** `SALT` hashes `(cpython, span)` by name (`njit.py:48`). A new semantics file that isn't added to the tuple leaves numba's disk cache serving **stale machine code with no error**.
3. **There is nowhere to put a function users call.** The rounding helper has no module. The planned `bare_round` check (DCD103) needs a fix hint that points at something real.
4. **Calls through a module attribute aren't fingerprinted by content.** `fingerprint` hashes the globals a step reads, but modules only by name. So `decimal.round(x)` in a step is keyed on the name `decimal`. If the function changes, the step's disk-cached kernel goes stale. This is from reading `fingerprint.py`; it hasn't been reproduced by running. It also applies to users' own helper modules, and to an external credit package.
5. **The expression language keeps its own function table.** It will drift from whatever library steps get.

## Design: two layers, one contract

### Layer 1: semantics (implicit, private), `_engine/compile/semantics/`

This layer makes operators and builtins, applied to a value inside a kernel, give **the same answer as the Python value would**. Users import nothing; they write ordinary Python.

```
_engine/compile/
  span.py                  the span type, its numba model, SpanArray boxing (the type, not its operations)
  money.py                 the cents type and its numba model (see "Money overloads"), after the spike
  semantics/
    __init__.py            imports every module (that registers the overloads); SALT over the package
    numbers.py             ← cpython.py: round, pow, ipow
    strings.py             ← span.py lines 113-250: ==, !=, len, startswith, endswith, in, []
    money.py               operators on the cents type, built on decider.ops.decimal
    dates.py               later, only if date arithmetic enters kernels
```

### Layer 2: functions users call (explicit, public), `decider.ops`

These are things CPython doesn't have, or doesn't do the way money and credit need. They are grouped by domain, and each one is callable from plain Python and from a kernel.

```
decider/ops/
  __init__.py              docstring: one line per domain; no re-exports (one canonical path)
  general.py               numeric helpers several domains need (clip, safe division, interpolation); grown on demand
  strings.py               explicit string functions, when one is needed (== / startswith / in are layer 1)
  decimal.py               rounding with a mode; scaled-int64 fixed point at any scale (money is scale 2)
  credit.py                per-record credit functions (credit-functions.md)
```

This takes the user's proposed domains (`credit`, `str`, `decimal`, `general`) with one change: `strings`, not `str`. `from decider.ops import str` would shadow the builtin in every file that does it. `from decider.ops import decimal` shadows the stdlib `decimal` in the same way. That is accepted, because stdlib `Decimal` can't be used inside a step anyway.

```python
from typing import Annotated
from decider import Money, param, step
from decider.ops import decimal
from decider.ops.decimal import Rounding

@step
def instalment(cents: Annotated[int, Money(cents=True)], rate: float = param(0.1)) -> Annotated[int, Money(cents=True)]:
    return decimal.round(cents * rate / 12, mode=Rounding.HALF_UP)
```

Each public function in `decider.ops` is written once with `@register_jitable`:
- It runs as Python in `interpreted` mode, in fallbacks and on the Python-per-row path.
- It is inlined into a kernel when a compiled step calls it.
- There is one source and no second implementation, and it doesn't need `@helper(signatures=)`.

### Rounding

`decider.ops.decimal.round(x, digits=0, mode=Rounding.HALF_UP)`, with `Rounding` an `IntEnum`. A mode is an ordinary int in a kernel, so one compiled function serves all three:

| Mode | Ties | Other values | Matches |
|---|---|---|---|
| `HALF_UP` | away from zero (2.5 → 3, −2.5 → −3) | nearest | `Decimal` `ROUND_HALF_UP`, the banking convention; the default for money |
| `HALF_EVEN` | to even (2.5 → 2, 3.5 → 4) | nearest | CPython `round()`, `Decimal` `ROUND_HALF_EVEN` |
| `DOWN` | toward zero | toward zero (2.9 → 2, −2.9 → −2) | `Decimal` `ROUND_DOWN` (truncation; not floor) |

- It works on the exact value of the double, like `cpython_round` today, so every mode agrees with Python. `2.675` is stored as `2.67499999…`, so `round(2.675, 2)` gives `2.67` under every mode. That is correct for the double, but it surprises people who wrote `2.675`. Money avoids it by staying in integer cents: the only floats being rounded are products like `cents * rate`, rounded to whole cents (`digits=0`). See question 3.
- `Money(cents=True)` gains a `rounding=` field (default `HALF_UP`) that the money overloads use. This extends an existing annotation rather than adding a new one.
- The `bare_round` check (DCD103) points at `decimal.round`.

### Money overloads

**The goal:** with `a, b` annotated `Money(cents=True)` and `rate` a float, `a + b`, `a - b`, `a * 3`, `a * rate`, `a / 4`, `a < b` and `sum(...)` do the right thing in every mode. That is, products and quotients round to whole cents by the field's `rounding`.

**Why money needs its own type.** A `Money(cents=True)` value is a plain `int64` in a kernel and a plain `int` in Python. Overloading `*` on `int64` would capture every integer in the process, which contract rule 2 forbids. So money gets what strings got, a decider-owned type:

- **Python side:** `Cents`, an `int` subclass with the operators. Everything that takes an int still takes it, and a step's money inputs arrive as `Cents` in every mode, including interpreted.
- **Kernel side:** `CentsType`, a numba type with an `int64` model. Money-annotated inputs and outputs are typed as it, and the operator overloads are registered on `CentsType` only, so nothing else is captured. `State` still stores plain `int64`; the type exists only at the kernel's edge.
- **The rules:**

  | Expression | Result |
  |---|---|
  | `Cents ± Cents` | `Cents` |
  | `Cents * int` | `Cents` |
  | `Cents * float` | `Cents`, rounded by the field's mode |
  | `Cents / int` and `Cents / float` | `Cents`, rounded by the field's mode |
  | `Cents / Cents` | `float` (a ratio) |
  | comparisons | as for int |
  | `Cents + int` | **a type error, not silent** (question 4) |

  The arithmetic goes through `ops.decimal`, so the explicit functions and the operators can't disagree.
- **int64 overflow** stays the checks' job (DCD101 `int64_overflow` knows money operands). The overloads don't add saturation.

**A spike comes first.** Before building, measure it the way `strings-span-ops.md` did for spans:
- the per-call boxing cost on `score()`;
- typing interactions in fused and packed control flow;
- params bundles that carry money;
- `assert_equivalent` across modes over `.xx5` ties.

If the spike shows a cost that the serving path can't afford, or typing traps, keep the explicit `ops.decimal` functions and drop the overloads.

### The contract (both layers, enforced by tests)

1. **Same answer in every mode.** That means interpreted, stepped, fused, `score()` and `Session`. Layer 1 matches the Python value's own behaviour; layer 2 matches its own Python body. Each function or overload has a test through `testing.assert_equivalent` over an edge corpus:
   - `.xx5` ties;
   - values around 2⁵³;
   - int64 bounds;
   - NaN, ±inf and ±0.0;
   - empty and non-ASCII spans.
2. **Overloads only where they can't capture anything else.** An overload is allowed on (a) a decider-owned numba type (`SpanType`, `CentsType`), or (b) a builtin at a **concrete** signature where the result becomes CPython's (`round(Float, Integer)`, `Float ** Integer`). Never on a generic type (bare tuples, every integer, every array). Only layer 1 may use `@overload` / `@overload_method` / `@intrinsic`; layer 2 uses `@register_jitable` only.
3. **Pure and nogil-safe.** No globals mutated and no Python objects. Serving kernels are `nogil=True` and run concurrently (`thread-safety.md`).
4. **Covered by the cache salt.** `SALT` hashes every file in `_engine/compile/semantics/`, `_engine/compile/{span,money}.py` and `decider/ops/`, found by walking them, not from a hand list.
5. **Docs follow `CLAUDE.md`.** A `decider.ops` function is public: it gets a one-line summary, the arguments that aren't obvious, and an example. Layer 1 is private: no docstring, just a module comment saying which behaviour it reproduces (as `cpython.py` has today).

### Where the other kinds go

| Kind | Home | Change |
|---|---|---|
| User helpers | `@helper(signatures=...)` stays the user API for their own kernel functions | Fix finding 4: `fingerprint` follows `module.attr` loads to the function and hashes it by content. Then `mylib.f(x)` is as safe as `f(x)`. That matters for any external credit package too. |
| Internal kernels (walker, matcher, scorecard, postfix, Arrow intrinsics) | stay with their owner | They may call `decider.ops`. They never register overloads; the Arrow intrinsics are the one allowlisted `@intrinsic` outside `semantics/`. |
| Expression-language functions | `steps/expr` | When `expr:` needs more than `min` / `max` / `abs`, its table maps names to `decider.ops` functions, not new opcodes. Don't change it before then. |

**Extension.** Third parties write `@helper` functions, or their own `@register_jitable` modules under the same contract. Overloads stay decider-only, because they change numba's typing for every user of the process (rule 2).

## Tests and checks

- **A convention test in `tests/test_conventions.py`:**
  - `numba.extending.overload*` and `intrinsic` appear only under `_engine/compile/semantics/`, plus the allowlisted Arrow intrinsics.
  - `register_jitable` appears only there, under `decider/ops/`, and in the internal-kernel modules.
  - Every file under those paths is covered by `SALT`; the test computes it and compares.
- **Checks:**
  - DCD103 `bare_round` points its `fix` at `decider.ops.decimal.round`, so the function lands **before** that check.
  - DCD102 `float_money` points at `Money(cents=True)` with the overloads, or at `ops.decimal` if the overloads were dropped.

## Where it lands in the plan

| Phase | Work |
|---|---|
| 1 | The convention test, with today's two loose modules allowlisted |
| 3 (with the `_engine` move) | `cpython.py` → `semantics/numbers.py`; the operations half of `span.py` → `semantics/strings.py`; `SALT` computed by walking. Mechanical, no behaviour change. |
| 5 | `decider/ops/` with `decimal.round` and its three modes; the `fingerprint` fix for `module.attr` calls (with a cache-miss test); **the money-overload spike**, written up in `notes/money-overloads.md` |
| 6 | Money overloads (if the spike says go) and `Money(rounding=)`; DCD103's fix hint; the rest of `ops.decimal` (scaled fixed point at any scale) |
| 7 | `decider.ops.credit` and the other credit homes (`credit-functions.md`) |

## Decisions (2026-10-05)

1. **Domains:** `credit`, `strings`, `decimal`, `general` (`strings`, not `str`).
2. **Decimal scale is per field:** `Money(scale=2)` extends `Money(cents=True)`. **Overflow:** float64 intermediates (exact to 2⁵³); int64 overflow stays the checks' job (DCD101).
3. **`round(2.675, 2)`:** exact-double semantics, `2.67` in every mode. Money stays in integer cents.
4. **`Cents + int` is a type error.** Write `Cents(n)` to say an int is cents.
