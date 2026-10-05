# Decimal and Currency instead of Cents

**Recommendation: build a `Decimal` value type (int64 mantissa, scale in the type) and `Currency` as
`Decimal` at scale 4. Drop `Cents`.** It reuses the money-overloads design (`money-overloads.md`,
verdict GO) with four things changed: the Python class is not an `int` subclass, the numba type is
keyed by scale instead of rounding mode, `Decimal ± int` is legal, and the rounding mode leaves the
type. Decimal128 (more than 19 digits) stays out of kernels in v1; it falls back to Python.

This supersedes decisions 2 and 4 of `pythonapi-redesign/kernel-functions.md`. Decisions 1 and 3
stand (exact-double `round(2.675, 2) == 2.67`; float64 intermediates; int64 overflow is the checks' job).

Measured with polars 1.41.2, numba 0.67.0, Python 3.14. Scratch scripts: `.scratch/decimal/`
(`pl1-5.py` polars, `spike.py` numba type, `divlu.py` 128-bit rescale). They are not committed.

## 1. What polars does

`D(p,s)` is a polars `Decimal(precision=p, scale=s)`. Observed, not read from docs.

| Expression | Result dtype | Observed |
|---|---|---|
| `D(18,4) ± D(10,1)` | `Decimal(38, 4)` | 3.1234 + 2.5 = 5.6234 |
| `D(10,2) + D(10,2)` | `Decimal(38, 2)` | precision 38 even when both inputs are small |
| `D(18,4) * D(10,3)`; `D(10,1) * D(10,3)` | `Decimal(38, 4)`; `Decimal(38, 3)` | scale is the **max**, not s1+s2 |
| `D(10,1) * D(10,1)`: 0.5·0.5, 1.5·0.5, 2.5·0.5 | `Decimal(38, 1)` | 0.2, 0.8, 1.2: exact product, **half-even** |
| `D(10,2) / D(10,2)`: .50/4, 1.50/4, 2.50/4, -1.50/4 | `Decimal(38, 2)` | 0.12, 0.38, 0.62, -0.38: half-even |
| `D / int` (Int64, UInt8, literal) | `Decimal(38, s)` | the int acts as a scale-0 decimal |
| `D op float` (Float64, Float32, literal) | **`Float64`** | `0.31 * 1.5` = 0.46499999999999997 |
| `D == / < D` (other scale), `D < int` | `Boolean` | exact |
| `D == / < float` | `Boolean` | decimal cast to Float64 first: `0.30 == 0.3` true, `0.30 == 0.1 + 0.2` false |
| `-D`, `abs`, `min`, `max` | same `(p, s)` | |
| `sum`, `cum_sum`; `mean` | `Decimal(38, s)`; `Float64` | |
| `cast D(10,2) → D(10,1)` of .15, .25, .35 | `Decimal(10, 1)` | 0.2, 0.2, 0.4: half-even |
| `cast Float64 → D(10,2)` of 2.675, 0.125, 0.375 | | 2.68, 0.12, 0.38: from the shortest repr, **not** the exact double (2.67) |
| `//`, `%`, `**` | error | unsupported for Decimal |
| `/ 0` | `ComputeError` | |
| result past 10^38 (`1e19 * 1e19` at scale 0) | `ComputeError: overflow in decimal multiplication` | never wraps |

What this means:

- **Every arithmetic result is Decimal128.** Operands are widened to precision 38 and the scale is
  `max(s1, s2)`. `*` and `/` compute the exact result and round it to that scale. This is not the SQL
  or Arrow rule (`s1 + s2` for `*`).
- **Rounding is half-even**, while decider's money default is HALF_UP. Ties differ: a frame-tier
  `0.5 * 0.5` gives 0.2 where a decider step gives 0.3 (question 3).
- **Where the user's "returns a Decimal" differs:** polars returns `Float64` for `Decimal * float`.
  We return a Decimal, so the float product is rounded back to the decimal's scale (section 4).
- **Overflow raises** past 38 digits. Compiled int64 kernels wrap instead (section 2).

## 2. Type model

| | Value |
|---|---|
| `Decimal` value | int mantissa `raw` plus `scale` (0 to 18), optional display `units`. A plain class, not an `int` subclass. |
| In a kernel | int64, typed `DecimalType(scale)`. Columns, `State`, params bundles and outputs stay plain int64. |
| Annotation | `Decimal[p, s]` (`Annotated[Decimal, ...]`): p up to 19, so that the tools see precision and scale |
| `Currency` | `Decimal[19, 4]`: 15 integer digits, 4 fractional. A subclass that only fixes the scale and takes `units=`. Operators return a plain `Decimal`. |

**Exact range** (`INT64_MAX = 9,223,372,036,854,775,807`, `INT64_MIN` reserved because `abs` wraps):

| Scale | Range |
|---|---|
| 2 | ±92,233,720,368,547,758.07 |
| 4 (`Currency`) | **±922,337,203,685,477.5807** |
| 6 | ±9,223,372,036,854.775807 |

A precision up to 18 always fits. Precision 19 is limited by int64, not by 10^19 - 1. So `Currency`
is "15 digits" only up to 9.22 x 10^14.

**Above 19 digits (Decimal128).** Tried in numba:

| Option | Finding |
|---|---|
| i128 as a numba type | `add`, `sub`, `mul` and compares are inline LLVM. **Division aborts the process**: `LLVM ERROR: Symbol not found: __udivti3`. Not an exception; nothing can catch it. |
| Register libgcc's `__udivti3` with `llvmlite.binding.add_symbol` | Works here on Linux. Depends on `libgcc_s` being loadable; other platforms differ. Not recommended. |
| Software 128-by-64 divide (`divlu`, uint64 only) plus one inline i128 multiply | Portable. 40,007 random and edge cases against Python ints: 0 mismatches. |
| Two-limb `Decimal128` | Needs 128x128 → 256-bit multiply and a 256-by-128 divide, and 16-byte column storage (numpy has no int128; Arrow decimal128 buffers are 16 bytes). Weeks, not days. |
| float64 for wide values | Loses exactness above 2^53. No. |

**Recommendation (minimum that works):**

- Kernels are 64-bit only. `Decimal[p, s]` with p above 19 raises at flow build with a message that
  says to cast in the frame tier. Compile-time failures already fall back to Python with a warning
  (`CLAUDE.md`), and Python ints are exact.
- The 128-bit intermediate is used **inside** `*`, `/` and compares of 64-bit values, which is where
  it is needed (a product of two 15-digit mantissas does not fit int64).

| 128-bit rescale `round_half_up(x * y / 10^k)`, 2M ops | Cost |
|---|---|
| int64 multiply + floor-divide (not rounded, wrong) | 2.3 ns |
| float64 multiply, divide, `rint` (not exact) | 2.5 ns |
| 128-bit multiply + `divlu`, product fits 64 bits (one branch) | 6.0 ns |
| 128-bit multiply + `divlu`, wide operands | 21 ns |

Precision is **not** in the numba type. The result of an op has the machine type
`DecimalType(max(s1, s2))` whatever the precisions. Precision lives in annotations, the boundary
cast (polars raises past `p`) and the checks, which compute the worst case (`max(i1, i2) + s` digits
for `±`, `i1 + i2 + s` for `*`) and flag anything above 19 (DCD101). One numba type per `(p, s)`
would buy only a compile-time refusal that makes `Currency + Decimal[18, 2]` (needs 20 digits)
fall back to Python for a value range nobody has.

## 3. JIT

Confirmed in `spike.py` (not an `int` subclass, scale in the type):

| Check | Result |
|---|---|
| `typeof`, `box`, `unbox` for the plain class | works; a `Decimal` object returns from a kernel |
| Python object crossing the dispatcher | 3.5 us against 0.3 us for an int; in a 2-tuple (a params bundle) **14 to 21 us against 0.4 to 0.6 us**. Same as `Cents`: bundles and columns stay int64, the type is made at the kernel edge by the wrap tag |
| Disk cache | `f(dec[4], dec[1])` is a hit in the second process |
| Mixed scales | `a + b`, `a * b`, `a / b` compile; `3.1234 + 2.5`, `* 2.5`, `/ 2.5` give 5.6234, 7.8085, 1.2494 (same as polars) |
| `x if c else y` with scales 2 and 4 | types as scale 4 (`unify` plus a `lower_cast` that multiplies) |
| `total = 0; for ...: total += d` | types as `Decimal` (int unifies with the decimal type). `Cents(0)` is no longer needed |
| `.5` ties | `0.10 * 0.05` = 0.01, `-0.10 * 0.05` = -0.01 (HALF_UP away from zero) |

Interpreted mode: the Python class implements the same arithmetic on Python ints (exact, so the
128-bit product is free) and the same float expression for comparisons with a float. `assert_equivalent`
over the corpus in section 8 is what proves they agree. `1.5 * Decimal(...)` reaches `__rmul__`
because `float.__mul__` returns `NotImplemented` for a non-int, non-float (checked), so the
spike's silent-float trap is gone.

## 4. Arithmetic rules

`s = max(s1, s2)`. A Python `int` operand acts as a scale-0 decimal and takes the other side's type.
"Rounds" always means HALF_UP, away from zero, applied once per operation (section 5).

| Expression | Result | How | Differs from polars |
|---|---|---|---|
| `D ± D` | `Decimal(s)` | scale the lower one up (exact) | precision (we do not widen) |
| `D ± int`, `int ± D` | `Decimal(s_D)` | the int is whole units | none. **Reverses decision 4** |
| `D * D` | `Decimal(s)` | exact 128-bit product, rescale by 10^min(s1,s2), rounds | HALF_UP, not half-even |
| `D / D` | `Decimal(s)` | 128-bit `raw1 * 10^(s + s2 - s1)` ÷ `raw2`, rounds. `ZeroDivisionError` | HALF_UP. Loses ratios: `Currency / Currency` has 4 decimals (question 4) |
| `D * int`, `int * D` | `Decimal(s_D)` | int64 multiply, exact | none |
| `D / int`, `int / D` | `Decimal(s_D)` / `Decimal(s_D)` | rounds | HALF_UP |
| `D * float`, `float * D` | **`Decimal(s_D)`** | `round(float(raw) * f)` exact-double, HALF_UP | polars gives `Float64` |
| `D / float` | `Decimal(s_D)` | same | polars gives `Float64` |
| `float / D` | `TypeError` ("use `float(d)`") | not needed | polars gives `Float64` |
| `D < D`, `==` etc. | `bool` | compare after scaling, in 128 bits (no overflow) | none |
| `D < int` | `bool` | exact | none |
| `D < float` (`==`, `!=`, ...) | `bool` | `float(raw) / float(10^s) < f` in both modes | same as polars (via Float64) |
| `sum(xs)`, `+=` | `Decimal(s)` | `+` rules; Python's `sum` starts at int 0, which now works; `iadd` is registered separately | none |
| `min(a, b)`, `max(a, b)` | `Decimal(max s)` | the kernel aligns scales; Python returns an operand, equal in value | none |
| `-D`, `abs(D)` | `Decimal(s)` | | none |
| `float(D)`, `int(D)` | `float`; `int` | `int` truncates toward zero | |
| `//`, `%`, `**`, `round(d, n)` | `TypeError` | `round` points at `decimal.round` (DCD103) | polars: unsupported |

- `float(raw)` is exact up to 2^53 (`raw` below 900,719,925,474.0992 at scale 4, R900 billion). Above
  that, `D * float` and `D < float` lose the low digits, as decided for float64 intermediates.
- The float comparison is the same expression in both modes, so it cannot disagree between them. It
  can disagree with the exact decimal value above 2^53.
- int64 overflow in `±` and `D * int` wraps in kernels and raises `OverflowError` in Python (a result
  outside int64 is refused rather than answered at a width the kernel does not have). It is
  not an equivalence case; DCD101 flags it.

## 5. Rounding

- **Mode is fixed at HALF_UP for operators.** It is not in the type, so the spike's traps 5 (mode
  unification), 7's mode part and 8 (`to_json` shape) disappear. `Decimal * float` and `D / D` round
  once to their result scale.
- **Where the field's `rounding=` applies:** storing a value into an annotated field of a **coarser**
  scale (a scale-4 result into `Decimal[18, 2]`) and `decimal.rescale`. This is a kernel-edge cast
  (output wrap), not a numba conversion. Modes: HALF_UP (default), HALF_EVEN, DOWN, as decided.
- **Double rounding:** `Currency * 0.0175` rounds to scale 4, and storing it into `Decimal[18, 2]`
  rounds again to scale 2. Polars rounds per operation the same way. Accept it; keep the whole
  computation at one scale (store at scale 4) when a single rounding matters.
- **`decimal.round`:** for a float it is the existing exact-double `round(x, digits, mode)`. For a
  `Decimal` it returns the same scale with the low digits rounded (polars `round(1)` keeps scale 2).
  `decimal.rescale(d, scale, mode)` changes the scale; `scale` must be a literal so the result type is
  known. Both are `@register_jitable` and share one HALF_UP body with the operators.
- Keep the stdlib comparison test (spike trap 9): `round_half_up` against `decimal.Decimal(x)
  .quantize(...)` for the three modes, negatives near zero included.
- **Constructor from a float:** `Currency(2.675)` is 2.67 (exact double, decision 3);
  `Currency("2.675")` is 2.68. Polars' cast gives 2.68 for the float (question 8).

## 6. Units and repr

| Option | Cost | Output for 4.5 / 1234.5 |
|---|---|---|
| (a) client supplies the string: `units="R"`, `"$"`, `"ZAR"` | none | `R 4.50`, `$ 4.50`, `ZAR 4.50` |
| (b) ISO code + `babel` | not installed; 2.18.0, BSD-3, **31 MB** of CLDR data, 29 ms import + 45 ms first format (`import decider` is already 0.9 s, so import time is minor; the install size and locale rules are not) | `format_currency(4.5, "ZAR", locale="en_ZA")` = `R4,50`; `en_US` = `ZAR4.50`; `1234.5` = `R 1 234,50` |

Babel answers a locale question nobody asked, and its South African output is not `R 4.50`. **Recommend
(a):** `units` is any string, shown as `f"{units} {amount}"`. No table, no dependency. If an ISO lookup
is wanted later, call `babel.numbers.get_currency_symbol` for the symbol only, imported lazily inside
the one function (question 7).

- repr of a `Decimal`: `Decimal('4.5000')`. Of a `Decimal` with units: `R 4.50` (trailing zeros trimmed
  down to two places, so `R 4.1234` stays). Units are display-only and excluded from `==` and `hash`.
- **Units never enter kernel types or cache keys.** A kernel value has none; the value that leaves a
  kernel is rebuilt from the int64 and the field annotation.
- **`ZAR + USD` is not an error at run time.** A runtime check would exist only in Python modes
  (kernels have no units), which breaks "same answer in every mode". Python keeps the left operand's
  units when both sides agree, else `None`. A static check on annotations (`Currency["R"]` against
  `Currency["$"]`) can do it later (question 6).
- Annotation spelling: `Currency` is `Decimal[19, 4]`; `Currency["R"]` adds display units as
  `FieldMetadata` (the debugger formats by it, as for `Money` today).

## 7. Mapping from today

| Today | After |
|---|---|
| `Money()` on a float (display `R`) | Stays as display metadata for a float amount. No alias for `Currency`. DCD102 already steers money away from float. |
| `Money(cents=True)`, `Cents`, `CentsType`, `Money(rounding=)` | **`cents=` removed outright** (never in a release; no shim). `Cents` is not built; `spike/money-overloads` stays as reference. |
| `Money(scale=2)` (kernel-functions decision 2) | `Decimal[p, s]`; amend that note |
| Boundary `_decimal_as_cents(money_scale=2)` truncating | Cast at the **annotation's** scale; the `money_scale` argument goes; truncation becomes the field's rounding. Same cost class: `to_physical().cast(Int64, strict=True)` is 9 ns/row, and out of range raises (`922337203685477.5808` at scale 4 fails). |
| Kernel int64 → frame column | `raw.cast(Decimal(38, s)) / Decimal(10^s)` (exact, verified at scale 4 for ±INT64_MAX): 64 ns/row |
| `decimal` columns of 20 to 38 digits | Fit int64 → cast works (existing `NeedsKernelSplit` text stays). Do not fit → strict cast raises, as today. |
| GUIDE line "`Money(cents=True)` is an int of cents" | Replaced by a `Decimal`/`Currency` section; its example runs in `tests/test_guide.py` |

Spike traps (`money-overloads.md`):

| Trap | Now |
|---|---|
| 1 wrap every source kind; `_drop` silently loses packing | **applies** (same wrap tag, same `FusedRunner` warning) |
| 2 `float op Cents` leaks a float | **gone** (no int base) |
| 3 `NotImplemented` falls through to `int` | **gone** |
| 4 `sum`, `+=`, `min`, `max` | `sum` works (`D + int` legal); `iadd` and `min`/`max` overloads **still needed** |
| 5 mode in the type must unify | **gone**; scale unification replaces it (verified) |
| 6 `return 0` does not type | **gone** (int unifies with `Decimal`) |
| 7 `T | None` needs `Optional(type)` | **applies** |
| 8 `to_json` shape | applies to the `units` metadata; omit defaults |
| 9 first `round_whole` wrong for negatives near 0 | **applies**; keep the stdlib comparison |
| 10 `@helper(signatures=)` has no money type | **applies**: `_numba_type` accepts `Decimal[p, s]` |
| new: i128 `udiv` aborts the process | use `divlu`; a test must call `*` and `/` with wide operands |
| new: `abs(INT64_MIN)` wraps | reserve `INT64_MIN` as invalid; boundary cast rejects it |

Checks (`pythonapi-redesign/checks.md`), wording only:

| Code | Change |
|---|---|
| DCD101 | "error when an operand is `Decimal`" (was `Money(cents=True)`); add the worst-case digits rule from section 2; `D * D` cannot overflow in its product, only in its result |
| DCD102 | "money held as float, `Money()` on a float"; `Decimal * float` is allowed and is not flagged |
| DCD103 | error when an output is `Decimal`; `fix` points at `decimal.round` |
| DCD110 | `float(d)` above 2^53 mantissa |

## 8. Work breakdown

| File | Work |
|---|---|
| `decider/ops/decimal.py` | `Decimal` (class, `[p, s]`), `Currency`, `Rounding`, `round`, `rescale`; Python operators (explicit `TypeError` for the rest); the `@register_jitable` `divlu` and HALF_UP helpers |
| `decider/_engine/compile/decimal_type.py` | `DecimalType(scale)`, int64 model, `typeof`/`box`/`unbox`, `lower_cast` (rescale up, int → decimal), `unify`, `can_convert_from`, `raw`/`wrap` intrinsics, the i128 multiply intrinsic. Not named `decimal.py`: a directory on `sys.path` would shadow the stdlib module |
| `decider/_engine/compile/semantics/decimals.py` | operators on `DecimalType` only, each with a one-line `TypingError` (cut down in `_why`) |
| `_engine/compile/{kernel,units,njit}.py` | wrap tag incl. `Optional`; probe and helper types; `SALT` covers the new files by walking |
| `_engine/run/runners/{interpreted,fused}.py` | wrap at bind; warn when a packed kernel is dropped |
| `decider/fields.py`, `types.py` | remove `cents`; `Currency["R"]` units metadata; plain-annotation of `Decimal[p, s]` |
| `_engine/boundary/dtypes.py` | scale-aware cast in, `raw / 10^s` cast out |
| `decider/checks/`, `GUIDE.md`, `kernel-functions.md` | wording above; a GUIDE section with a runnable example; mark decisions 2 and 4 superseded |

Tests (`tests/ops/`, `tests/compile/`, `tests/boundary/`):

- `testing.assert_equivalent` over a corpus: `.xx5` ties (positive and negative) for `*`, `/`, float
  products, `rescale`; scales (2, 2), (2, 4), (4, 0); values near ±2^53 raw and near the int64 bounds
  (compare and `divlu` wide path, not overflow); `x if c else y` with two scales; `total = 0` loops.
- The stdlib `decimal` comparison for all three modes; the `divlu` check against Python ints.
- `packed` contains a decimal branch and loop; plain int and `(1, 2) == (1, 2)` unchanged; a changed
  overload changes `SALT`.
- `Currency + 1`, `Currency * 1.5`, `1.5 * Currency(3)` (a `Decimal`, 4.5 = 45000 at scale 4);
  `float / D` and `D // D` are a `TypeError` in Python and a clear message from `strict_compile=True`.
- Boundary: Decimal column in and out at scales 2 and 4, out of range, `INT64_MIN`, null.
- Cost: warm `score()` on the three-step flow from `money-overloads.md` with `Currency` against
  plain int cents: p50 and p99 within noise (spike: 32.4 against 32.6 us fused); batch rows/s; first
  call with a cold cache.

## 9. Open questions

1. **Decimal128 in kernels?** Recommend no for v1: `Decimal[p, s]` with p above 19 raises at build,
   wide values go through the frame tier or the Python fallback. Say if a real flow needs more than
   15 integer digits at scale 4.
2. **Numba type keyed by scale only** (precision lives in annotations and checks)? Recommend yes.
   The alternative, one type per `(p, s)`, makes `Currency + Decimal[18, 2]` fall back to Python.
3. **Operator rounding is HALF_UP; polars is half-even.** Ties differ by one unit at the last place
   between a frame-tier calculation and a step. Recommend keeping HALF_UP, and a note in the GUIDE.
4. **`Decimal / Decimal` returns a `Decimal`** at the larger scale (polars), so `fee / income` has
   4 decimals. Recommend following polars and using `float(a) / float(b)` for ratios.
5. **`Decimal ± int` is legal** (reverses decision 4, since no unit risk remains). Recommend yes.
6. **Unit mismatch** (`ZAR + USD`): not a runtime error; Python keeps equal units, else drops them;
   a static check later. Recommend that.
7. **Units:** client-supplied string, no library. Recommend; add an ISO lookup only on request.
8. **Float constructor:** `Currency(2.675)` is 2.67 (exact double) or 2.68 (shortest repr, as
   polars casts)? Recommend 2.67 and documentation pointing at strings.
9. **Python-mode overflow:** `Decimal` raises `OverflowError` outside int64 while kernels wrap
   (DCD101 flags). Recommend this; the alternative is wrapping in Python too, which hides it.
10. **`Money`:** remove `cents=`, keep `Money()` as display for floats, no alias. Or remove `Money`
    entirely once `Currency["R"]` drives the debugger formatting. Recommend the first now, the
    second when the debugger reads units from `Currency`.

## 10. Decisions (2026-10-05)

- **Operator rounding is half-even, matching polars** (question 3 reversed). Explicit rounding keeps every mode: `decimal.round(x, scale=1, mode=Rounding.HALF_UP)` gives banking rounding when a step wants it.
- **`Decimal ± int` is legal**; the int is whole units (question 5).
- **Units are a client-supplied display string** (`units="R"`); no Babel (question 7).
- Questions 1, 2, 4, 9 and 10 take the recommendations above.
- **Still open:** question 8, `Currency(2.675)` from a float (2.67 exact double, or 2.68 shortest repr as polars casts). Settle before building.
