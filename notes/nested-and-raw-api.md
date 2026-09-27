# What a user should write for strings and nested data

Measured on 2026-09-26 at `d825f31`, loaded dev box, numba 0.67.0, polars 1.41.2,
CPython 3.14.5. The probes were written to `.scratch/`, which is gitignored, so
they are not in the tree: every figure and every message below is quoted from a
run, and §2 and §4 say what each probe did well enough to rebuild it.

## Recommendation first

**Delete `Raw`, `Rows` and `raw_str` from the public surface. The plain Python
type is the whole annotation surface; the representation is the engine's choice,
made per call from `n`.** `TypedDict` already means "fixed-schema struct" — keep
that and honour it further. Spend the effort on the JSON boundary instead: it is
where every measured silent-wrong-answer lives.

The argument is not taste. It is three numbers:

1. A representation annotation cannot be right, because the right representation
   depends on `n`, and the same pipeline serves `score()` (n=1) and `run()`
   (n=4.1M). `notes/strings.md` already shows `Raw[str]` winning 4.7x in batch
   and losing 2x on a single record. The user does not know `n` when they write
   the annotation. The engine knows it at every call.
2. `Rows[Item]` loses on **both** axes at every size measured (below). It is not
   a trade-off, it is a cost.
3. Nobody wrote any of them. `Raw[`, `Rows[`, `raw_str` appear **zero** times
   across the 24 real projects in `example_projects/` and `projects/`; the only
   uses in the repo are 14 lines in two test files. They are exported from
   `decider/__init__.py` and appear **0** times in `GUIDE.md`. There are no users
   to break.

---

## 1. Inventory: every annotation that changes representation today

Batch / `score()` figures for strings are from `notes/strings.md`; the `Rows` and
`list[dict]` figures are new (§2).

| What a user writes | Representation in a kernel | Batch | `score()` p50 | Joins the fused kernel? |
|---|---|---|---|---|
| `float` / `int` / `bool` | float64 / int64 / bool | 173M | 22.2 µs | yes |
| `T \| None` | the array plus a validity mask, typed `Optional(T)` | — | — | yes |
| `= missing_as(x)` | the array with `x` written over nulls | — | — | yes |
| `str` | numba `unicode_type`, one compiled call per row | 0.50M | 26.7 µs | no — `Fallback` backed by a dispatcher |
| `str \| None` | `Optional(unicode_type)` | — | — | no |
| `bytes` | nothing: refused, runs in plain Python | — | — | no — plain-Python `Fallback` |
| `Raw[str]` | int32 dictionary code | 2.3M | 56.3 µs | **yes** |
| `Raw[bytes]` | `(address, byte length)` span into Arrow memory, `-1` for null | 7.0M | 104.5 µs | **yes** |
| `-> str` | — | — | — | no |
| `-> Raw[str]` | int32 code | — | — | yes |
| `-> bytes` | (accidentally compiles; see H11) | — | — | yes |
| `list[dict]`, `list[float]`, `list`, `dict`, a bare `TypedDict` | nothing: plain Python per row | 0.13M | 29.4 µs | no |
| `Rows[Item]` | one flat float64/int64/bool array per `Item` field, sliced per parent row | 0.06M | 154.3 µs | no — compiled, one call per row |
| `Table` (`param_table`) | one read-only array per column, in every mode | — | — | yes (params, not request data) |

`Table` is a **params** construct, documented and working. It is out of scope
below; keep it exactly as it is.

### What each one forbids, and the real message

Every message below is copied from a run.

`Rows[Item]` refuses any field that is not `float`/`int`/`bool` — including the
`str` field the question asks about, and including a nested list:

```
TypeError: Rows[Item] field(s) ['sku'] must be float, int or bool for now
TypeError: Rows[Item] field(s) ['tags'] must be float, int or bool for now
```

It names `Rows[Item]`, not the user's class, and offers no fix. **There is no way
today to say "this field of my Item is a string."**

Semantic `bytes` is refused with a good message:

```
reads 'value' as bytes: numba can't type a scalar bytes value, only a byte array,
so no kernel can compare it whole
```

Everything else degrades with a warning rather than a refusal:

```
reads 's' as str, which no fused kernel holds
writes 'out_str' as str, which no fused kernel stores
reads 'items' as Rows[...], which runs one call per row, outside the shared kernel
reads 'items' as list[dict], which no kernel takes
reads 's' as <class 'datetime.date'>, which no kernel takes
```

### Hazards, all reproduced

**H1 — a `Raw[str]` step that falls back returns the wrong answer, silently.**
The worst finding. `stepped.py:109-112` sends a plain-Python `Fallback` *Python*
values — semantic strings, not codes. So any `Raw[str]` step that fails to
compile for an unrelated reason receives a `str` and compares it against an int:

```python
PRIORITY = raw_str("priority")

def helper(x):                       # undeclared -> the step runs in Python
    return x

def raw_but_uncompilable(s: Raw[str]) -> bool:
    return helper(s) == PRIORITY
```

```
interpreted  -> [True, False]
fused        -> [False, False]       # no error, no hint that the value changed type
```

The only trigger needed is anything numba cannot compile — an undeclared helper,
a generator inside `any()`, an `import` in the body. `raw_str()` called *in* the
body is the same bug arriving by a shorter route (it is itself uncompilable).
`assert_equivalent` **does** catch it, which is the one mitigation:

```
AssertionError: run() in stepped mode differs from interpreted: DataFrames are
different (value mismatch for column "raw_but_uncompilable")
```

**H2 — `Raw[str]` has no usable fill.** `missing_as("none")` raises with no step
name, no input name, and is *silently wrong* in interpreted mode:

```
interpreted  -> [False, False]       # wrong: row 0 is "priority"
stepped      RAISED ValueError: invalid literal for int() with base 10: np.str_('none')
```

`missing_as(0)` works but means "the code 0", which the user cannot see is
whatever string happened to be interned first. There *is* a good message for
this case in `extract.py::_fill_for`, but it is unreachable from the engine:
`State.from_frame` only routes `float`/`int`/`bool` through `extract_frame`
(`state.py:199`, `TYPED = (float, int, bool)`). Reached directly it says:

```
ValueError: input 's' is a `str` column with a string fill 'none'; a `str` input
enters the kernel as a dictionary code, so its fill must be an int code.
```

**H3 — `Raw[bytes]` has no null escape at all.** A required null gives advice
that cannot be followed:

```
MissingInputError: input 's' of step 'p/raw_bytes_null' is required but column 's'
has 1 null row(s) of 2. Fix the data, or declare `s: T = missing_as(fill)` or `s: T | None`.
```

but `missing_as(b"")` → `ValueError: invalid literal for int() with base 10: b''`,
`missing_as("")` → the same, and `Raw[bytes] | None` **disagrees between modes**
(H4): `interpreted` raises `TypeError: '>=' not supported between instances of
'NoneType' and 'int'` while `stepped`/`fused` return `[True, False]`. Both exits
from the message are closed.

**H5 — `Rows[Item] | None` breaks on `run`.** `AttributeError: 'list' object has
no attribute 'price'` — the optional path hands the step the raw Python list, not
the namedtuple. `score` happens to work.

**H6 — `Rows[Item] = missing_as([])` is broken everywhere.** `AttributeError`
interpreted, `ValueError: cannot compute fingerprint of empty list` compiled.
This is the *first thing a real user writes*: `income.py:117` in project 00 is
`variable_pay_history: list[float] = missing_as([])`, and F7 records three
projects hitting the list-fill path.

**H7 — a JSON `null` inside a `Rows[Item]` field becomes `NaN`, silently.**
`build_rows` assigns `item[name]` into a float64 array, so `None` becomes `nan`
and poisons the total. The plain type raises:

```
Rows[Item]   null price  -> [nan, 2.0]
list[dict]   null price  RAISED TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
```

Opting into the annotation turns a loud failure into a quiet wrong number, on a
money path. This is the hazard to weigh above all the others.

**H8 — `Raw` and `Rows` accept anything and mean nothing outside two cases.**
All silently accepted, all no-ops: `Raw[Item]`, `Raw[list[dict]]`, `Rows[dict]`
(a zero-field schema, because `get_type_hints(dict)` is `{}`), `-> Rows[Item]`,
and `-> Raw[bytes]` (which returns a dangling span as a two-element list column).
A family whose members are accepted everywhere and honoured in two places is not
a family.

**H9 — `raw_str()`'s codes are process-order dependent** (already noted as item 7
of `notes/review-gap-fixes-branch.md`). A stored decision cannot be replayed
against a code.

**H11 — `-> bytes` compiles into a shared `Kernel` and returns real `bytes`**,
with no fallback and no warning, while `-> str` correctly becomes a `Fallback`.
`njit.py:110-113` checks `str` outputs only. It gives right answers on the cases
probed, which makes it worse, not better: it is unreviewed surface.

**H12 — a `list[float]` output whose branches differ in element type crashes with
an empty message and does not fall back.** `return [1, 2] if x < 1 else [1.5]`
raises a naked `AssertionError` from `numba/cpython/listobj.py:1131`.
`FALLBACK_ERRORS` is `(NumbaError, UnsupportedBytecodeError)` plus
`NotImplementedError`, so numba's own `AssertionError` escapes. One-line fix for
whoever owns compile fallback; it is F8's remnant, now a crash instead of silent
truncation.

---

## 2. Measured: `Rows[Item]` is a net loss at every size

One step summing `price * qty` over a list of items, `fused`, 3000 `score()`
calls, against the same step written `list[dict]`.

| items/record | rows | `list[dict]` batch | `Rows` batch | `list[dict]` p50 | `Rows` p50 |
|---|---|---|---|---|---|
| 5 | 200k | 0.13M/s | 0.06M/s | **29.4 µs** | 154.3 µs |
| 20 (20x heavier body) | 50k | 0.02M/s | 0.02M/s | **77.0 µs** | 147.7 µs |
| 200 | 20k | 0.01M/s | ~0.00M/s | **57.8 µs** | 182.2 µs |

Numeric ceiling for reference: 129M rows/s, 21.7 µs.

The cause is `build_rows`, timed against the Python loop it replaces:

| items | `build_rows` p50 | the loop it replaces |
|---|---|---|
| 1 | 17.33 µs | 0.37 µs |
| 5 | 21.27 µs | 1.03 µs |
| 20 | 23.82 µs | 3.36 µs |
| 200 | 87.55 µs | 34.32 µs |

`build_rows` is three nested Python loops with a numpy scalar store per cell
(`rows.py:69-75`), so it is *strictly* more work than the user's loop, at every
size. Its 17 µs floor at one item is roughly the entire numeric `score()` budget:
the annotation doubles single-record latency before the body runs. There is no
size or body weight at which the current implementation pays.

That is a fact about `build_rows`, not about the idea. A vectorised conversion
straight off the Arrow list-of-struct buffer (`pl.Series.struct.field(...)`, no
Python loop) could plausibly win in batch. It still would not win at n=1, for the
same reason `Raw[str]` does not: there is nothing to amortise.

---

## 3. The proposal

### The rule

> **An annotation says what the value means. It never says how it is stored.**
> The engine picks the representation per call, from `n` and from what the step
> does with the value.

This is the one idea. Everything else follows.

It is forced by the workload: 90% of calls are `score(dict)` with n=1, the same
pipeline also runs 4.1M-row batches, and `notes/strings.md` shows the best
representation differs by 2-4x *in opposite directions* between those two. An
annotation is fixed at authoring time; `n` is not known until the call. So an
annotation that names a representation is guaranteed wrong on one of the two
paths, and the user has no way to fix it.

### The surface

| What the user has | What they write |
|---|---|
| a number, a flag | `float`, `int`, `bool` |
| a string | `str` |
| raw bytes | `bytes` |
| a date | `date`, `datetime` |
| a null with a fallback | `= missing_as(x)` |
| a null the step handles | `T \| None` |
| a list of numbers | `list[float]`, `list[int]` |
| a list of records | `list[Item]`, `Item` a `TypedDict` |
| one nested record | `Item`, a `TypedDict` |
| a free-form map | `dict` |
| policy rows, retuned per call | `Table = param_table({...})` |

Nine spellings, all stdlib, zero new names. `Raw`, `Rows` and `raw_str` are gone.

### Answers to the specific questions

**Is `Raw[T]` the right family name across `str`, `bytes`, `list[dict]`, `dict`?**
No, and neither is naming the shape. `Raw` is the wrong *kind* of word: it names
a representation, which is the one thing the annotation must not do. Empirically
it has also failed as a family — `Raw[Item]`, `Raw[list[dict]]` and `Raw[int]`
are all accepted and all mean nothing (H8), and the value a `Raw[str]` step
receives changes type depending on whether the step happened to compile (H1), a
fact invisible in the user's source.

**Does a `TypedDict` annotation alone suffice to mean "fixed-schema struct"?**
Yes — and it is already doing that job, which is the strongest evidence in this
note. `parse.py::has_date` recurses into a `TypedDict` and **not** into `dict`
(measured: `has_date(list[Account])` is `True`, `has_date(list[dict])` is
`False`), and `dummy` builds one field by field. So `TypedDict` is already the
only way to get JSON date strings coerced inside nested data. Four unrelated
projects discovered this independently and wrote the same comment:

> `# Typed so a served JSON request's account date strings arrive as dates.`

— `03/sonnet/loan_granting/affordability.py:139`,
`05/sonnet/business_nested/structure.py:207`,
`06/sonnet/consolidation/inventory.py:176`,
`11/sonnet/business_credit_e2e/covenant.py:221`.

A user who has already written `class Account(TypedDict)` to fix their dates
should not then have to write `Rows[Account]` to go fast. One concept, already
stdlib, already in their file.

**How does a user say "this field of my Item is a string"?** They write
`sku: str` in the `TypedDict`, and it works, because the item is a Python dict.
Today `Rows[Item]` refuses exactly that (`must be float, int or bool for now`) —
which is the annotation's limitation, not the field's. Remove the annotation and
there is nothing to refuse.

**What composes, and what is refused?**

Composes, and must keep working:

- a `TypedDict` with any field types, including `str`, `date`, a nested
  `TypedDict`, `list[Item]`, `list[float]`
- `list[Item] | None`, `list[Item] = missing_as([])`, `Item | None`
- `list[Item]` as an **output**, and read back by the next step

Refused, loudly, at bind time:

- `Raw[...]` and `Rows[...]` — during the transition, a `DeprecationWarning`
  naming the plain type; afterwards a `TypeError`
- a `TypedDict` field whose type the boundary cannot coerce (an arbitrary class):
  name the class, the field and the owning `TypedDict`
- `Rows[dict]` / `Raw[Item]` and friends: gone with the annotations

### Rejected alternatives

**A. Keep `Raw[T]` as the family; extend it to `Raw[list[Item]]`, `Raw[Item]`.**
Rejected: it entrenches the rule this note argues against, it has already failed
as a family (H8), and it needs a fifth, sixth and seventh set of fill/null rules
when the existing two are broken (H2, H3).

**B. Name the shape: `Rows[Item]` for lists, `Struct[Item]` for one record,
`Raw[T]` for scalars.** Rejected: three concepts to learn for a measured
*negative* return (§2), and `Struct[Item]` is pure ceremony — `Item` alone
already carries the schema. It also still leaves H1 (the fallback changes the
value's type) untouched.

**C. `TypedDict` plus an explicit array opt-in (`@step(arrays=True)`).**
Rejected as premature: there is nothing to opt into. Revisit only if a vectorised
`build_rows` beats the Python loop — and at that point the switch is a batch-size
threshold, not an annotation.

**D. Keep `Raw[str]` but make the Python fallback convert to codes too.**
Rejected as insufficient. It fixes one hazard of ten and leaves an annotation
that is 2x slower on 90% of calls. The work is the same size as making the engine
choose, so choose.

**E. `Enum` for a closed vocabulary.** The one alternative worth keeping on the
shelf. It is *semantic* ("this value is one of these"), not representational, so
it does not violate the rule; it legitimises a dictionary code because the codes
are then known at build time and stable across processes, which fixes H9 and
makes a stored decision replayable. `DecisionTableConfig` already requires this
(`GUIDE.md`: "A String *output* of a parameter table must be an `Enum` column, so
its values are known up front"). Parked because plain `str` plus engine-chosen
coding gets the same speed with no annotation; revisit for replay stability, not
for speed.

### What the engine should do instead

The information `Raw[str]` carried by hand is all available without it:

- `njit.py` already knows a step reads a `str` and can see whether the body only
  compares it (`==`, `!=`, `in`) against a literal or a `str` param. That is the
  case `Raw[str]` exists for, and it is exactly the case where coding is sound.
- The runner already knows `n`. `notes/strings.md` puts Python-per-row within
  5 µs of the numeric ceiling at n=1 and 4.7x behind codes at n=200k, so the
  switch is a row-count threshold, measured once, not a user decision.
- `exe.fallbacks()` already reports what did not make the kernel, so a batch user
  who cares can see it without annotating anything.

---

## 4. The guide section I would ship

Drafted here, not in `decider/GUIDE.md` (others are editing that file). Every
block below was run verbatim against `d825f31` and passes. The table after them
adds three promises this design makes that do **not** hold yet, so they are not
in the shipped text. Single record first throughout.

> ### Strings and nested data
>
> Write the plain Python type. A string is `str`, a list of records is
> `list[Item]` with `Item` a `TypedDict`, a nested object is that `TypedDict`,
> and a free-form map is `dict`. decider picks how to store each one; you never
> annotate a representation, because the best one differs between a single
> record and a 4-million-row batch and decider knows which it is running.
>
> A `TypedDict` is worth writing whenever the shape is fixed: it is what lets a
> served JSON request's date strings arrive as `date` objects inside nested data.
>
> ```python
> import datetime as dt
> from typing import TypedDict
>
> import polars as pl
> from decider import Engine, flow, missing_as, param
>
> class Account(TypedDict):
>     balance: float
>     instalment: float
>     opened_date: dt.date
>
> def exposure(accounts: list[Account] = missing_as([])) -> float:
>     return float(sum(a["balance"] for a in accounts))
>
> def band(bureau_score: float, cutoff: float = param(600.0)) -> str:
>     return "A" if bureau_score >= cutoff else "B"
>
> def approved(exposure: float, band: str, cap: float = param(50000.0)) -> bool:
>     return band == "A" and exposure <= cap
>
> pipeline = flow(exposure, band, approved, name="credit").emit("exposure")
> exe = Engine().bind(pipeline, mode="fused")
> accounts = [{"balance": 8500.0, "instalment": 450.0, "opened_date": dt.date(2022, 3, 1)}]
> assert exe.score({"bureau_score": 700.0, "accounts": accounts})["approved"] is True
> assert exe.score({"bureau_score": 700.0})["exposure"] == 0.0      # the key was absent
>
> df = pl.DataFrame({"bureau_score": [700.0, 500.0], "accounts": [accounts, []]})
> assert exe.run(df)["approved"].to_list() == [True, False]
> ```
>
> Nulls work the same for nested data as for a number: `= missing_as([])` fills
> an absent or null column, and `| None` lets the null through.
>
> ```python
> def oldest_days(accounts: list[Account] | None, decision_date: dt.date) -> float:
>     if accounts is None:
>         return -1.0
>     return float(max((decision_date - a["opened_date"]).days for a in accounts))
>
> exe = Engine().bind(flow(oldest_days, name="credit"), mode="fused")
> assert exe.score({"accounts": accounts, "decision_date": dt.date(2026, 1, 15)})["oldest_days"] == 1416.0
> assert exe.score({"accounts": None, "decision_date": dt.date(2026, 1, 15)})["oldest_days"] == -1.0
> ```
>
> One nested record is the `TypedDict` itself; a map whose keys are data, not
> schema, is a plain `dict`.
>
> ```python
> class Expenses(TypedDict):
>     accommodation: float
>     food: float
>
> def declared_expenses(expenses: Expenses) -> float:
>     return expenses["accommodation"] + expenses["food"]
>
> def total_declared(expenses: dict) -> float:
>     return float(sum(expenses.values()))
>
> record = {"expenses": {"accommodation": 3500.0, "food": 1800.0}}
> assert Engine().bind(flow(declared_expenses, name="p"), mode="fused").score(record)["declared_expenses"] == 5300.0
> assert Engine().bind(flow(total_declared, name="p"), mode="fused").score(record)["total_declared"] == 5300.0
> ```
>
> A step may also **write** a list of records, and the next step may read it
> back. Write one row per thing you are describing; don't return parallel lists.
>
> ```python
> from decider import step
>
> @step(output="per_account")
> def split(accounts: list[Account]) -> list[dict]:
>     return [{"ref": i, "share": a["balance"]} for i, a in enumerate(accounts)]
>
> @step(output="total")
> def total(per_account: list[dict]) -> float:
>     return float(sum(d["share"] for d in per_account))
>
> exe = Engine().bind(flow(split, total, name="p").emit("per_account"), mode="fused")
> assert exe.score({"accounts": accounts})["per_account"] == [{"ref": 0, "share": 8500.0}]
> assert exe.score({"accounts": accounts})["total"] == 8500.0
> ```
>
> Strings compare as strings in every mode, including against a `str` param, and
> a step may return one. A step reading or writing a `str` runs one compiled call
> per row instead of joining the fused kernel; `exe.fallbacks()` says so. On a
> single record that costs about 5 µs, so leave it alone unless you are running
> millions of rows.
>
> ```python
> def is_priority(channel: str, flagged: str = param("priority")) -> bool:
>     return channel == flagged
>
> def grade(bureau_score: float) -> str:
>     return "A" if bureau_score >= 700 else "B"
>
> exe = Engine().bind(flow(is_priority, grade, name="p"), mode="fused")
> assert exe.score({"channel": "priority", "bureau_score": 720.0}) == {
>     "channel": "priority", "bureau_score": 720.0, "is_priority": True, "grade": "A"}
> ```

### Which examples do not work yet

Each block above was run as written; the table records which promises they rest on.

| # | Example | Status | Depends on |
|---|---|---|---|
| 1 | `list[Item]` in, `score` + `run`, `missing_as([])` | **works** | — |
| 2 | `list[Item] \| None` | **works** | — |
| 3 | a `TypedDict` input | **works** | — |
| 4 | a bare `dict` input | **works** | — |
| 5 | `list[dict]` **output**, `score` + `run` | **works** | — |
| 6 | `list[float]` with `missing_as([])` | **works** | — |
| 7 | `str` input vs `str` param, every mode | **works** | — |
| 8 | a `str` output | **works** | — |
| 9 | a null inside an item raises, naming the field | **broken** | the nested-data work (§5 P1) |
| 10 | a missing key inside an item raises, naming the field | **broken** | the nested-data work (§5 P1) |
| 11 | a fractional float into an `int` input is refused | **broken** | the JSON-boundary work (§5 P2) |
| 12 | date strings inside `list[Item]` coerced | **works** | — |

Nine of twelve already run. The three that do not are **boundary validation**,
not representation features — no new annotation would fix any of them. Today:
example 9 gives `TypeError: unsupported operand type(s) for +: 'int' and
'NoneType'` (and a silent `nan` if the step is `Rows[Item]`), example 10 gives a
bare `KeyError: 'balance'`, example 11 silently answers `24` for
`{"term": 12.5}`.

Worth recording separately: the nested **output** side, which four real projects
worked around with parallel primitive lists (00, 02, 06, 11 — see
`example_projects/00-shared-credit-core/sonnet/NOTES.md:224-249`), is fixed at
this commit. `list[dict]`, `list[Item]`, `list[float]`, `list[str]`,
`list[date]`, `dict`, `list[list[float]]`, `frame_step` outputs and chaining a
`list[dict]` column into the next step all work in `score` and `run`. The
struct-of-arrays workaround should be deleted from those projects and the guide
should say so.

---

## 5. The JSON boundary

A record arrives as `json.loads(body)` → `coerce_record(record, dates)` →
`score(dict)`. `coerce_record` (`serving/parse.py:48`) validates **only** inputs
whose annotation contains a `date`; everything else is passed through as JSON
sent it, and lands in `np.array([...], dtype)` in `engine.py:279`, which is what
actually does the coercion.

### What happens today

| JSON value | Declared | Today | Verdict |
|---|---|---|---|
| key absent | `float` | `MissingInputError: input 'x' of step 'p/req' is required but column 'x' is not in the input frame or record. Fix the data, or declare ...` | good |
| `null` | `float` | same error, "has 1 null row(s) of 1" | good |
| absent or `null` | `float \| None` | `None` | good |
| absent or `null` | `= missing_as(0.0)` | the fill | good |
| `1000` | `float` | `1000.0` | good |
| `1000.5` | `int` | **`1000`**, silently truncated | **wrong** |
| `-1.9` | `int` | **`0`** | **wrong** |
| `9223372036854775807` | `int` | **`-9223372036854775808`** | **wrong** (known; wraps) |
| `2**63` | `int` | `OverflowError: Python int too large to convert to C long` | unnamed |
| `"1000"` | `float` | `1000.0`, silently coerced | tolerable, undocumented |
| `true` | `float` | `1.0` | tolerable, undocumented |
| `"abc"` | `float` | `ValueError: could not convert string to float: 'abc'` | no input or step name |
| `[1.0]` | `float` | `TypeError: Can only insert i64 at [0] in [2 x i64]: got double` | opaque |
| `{}` | `float` | `TypeError: float() argument must be a string or a real number, not 'dict'` | no names |
| item missing a key | `list[dict]` or `Rows[Item]` | `KeyError: 'price'` | no step, input, index or field |
| item field `null` | `list[dict]` | `TypeError: ... 'int' and 'NoneType'` | loud, unnamed |
| item field `null` | `Rows[Item]` | **`nan`** | **silent, wrong** |
| item field `1000` where `float` | either | `1000.0` | good |
| item has extra keys | either | ignored | good |
| `items: []` | `Rows[Item]` | `0.0` | good |
| `items: null` | `Rows[Item]` | `MissingInputError` recommending `missing_as(fill)` / `T \| None` — **both broken for `Rows`** (H5, H6) | **dead-end advice** |
| `"2022-03-01"` inside `list[Account]` | `TypedDict` with a `date` field | coerced to `date(2022, 3, 1)` | good |
| `"2022-03-01"` inside `list[dict]` | `list[dict]` | left a `str` → `TypeError: unsupported operand type(s) for -: 'str' and 'str'` in the body | correct-by-design, needs documenting |

One good result worth recording: **a `list` input is a `list` on both paths now.**
`score()` and `run()` both hand the step a Python `list`, and the ordinary
`xs or []` idiom works. F3g ("the Python type of a `list` input depends on the
entry point", which forced project 00 to write `is None` everywhere) is fixed at
this commit.

### What the surface must promise

1. **A declared type is coerced, or refused by name — never silently truncated.**
   `12.5` into an `int` must raise naming the input and the step. This is money.
2. **A null follows the same three rules everywhere.** Top level or inside a
   `TypedDict` field: required → error, `| None` → `None`, `missing_as` → fill.
   The `nan` in the table above is the promise being broken today, and it is the
   single strongest argument for deleting `Rows[Item]`: opting in silently
   converts a loud failure into a quiet wrong number.
3. **Every boundary error names the step, the input, and for nested data the
   index and the field.** Six rows above name nothing. A `KeyError: 'price'` from
   a 40-account request is not debuggable.
4. **`missing_as` and `| None` work for every declared type, or the
   `MissingInputError` stops recommending them.** They are currently broken for
   `Rows[Item]` and `Raw[bytes]` while being recommended for both.
5. **`score(dict)`, `run(frame)` and all three modes agree.** `GUIDE.md` already
   promises this. `Raw[bytes] | None` and `Rows[Item] | None` break it.
6. **A `TypedDict` field is validated and coerced; a bare `dict` is passed
   through untouched.** This is already true for dates and is the right line —
   write it down, so `dict` is understood as the explicit opt-out.

Promises 1-4 are the whole remaining cost of this design. None of them needs a
new annotation.

---

## 6. Migration cost

**Measured blast radius: zero users.**

- `Raw[`, `Rows[`, `raw_str`: **0** occurrences across `example_projects/` (24
  shipped projects) and `projects/loan_scoring`.
- **0** occurrences in `decider/GUIDE.md`. They were exported
  (`decider/__init__.py:48`, and in `__all__`) but never documented.
- In-repo uses: 10 lines in `tests/run/test_compiled_fallback.py`, 4 in
  `tests/run/test_rows.py`.

**Non-breaking path.** Keep the names one release as deprecated aliases:
`Raw[str]` → `str`, `Raw[bytes]` → `bytes`, `Rows[Item]` → `list[Item]`, with a
`DeprecationWarning` naming the replacement. `types.py` already normalises
through `raw_base`/`rows_item`, so the alias is roughly ten lines in
`base_annotation`. Then remove.

Two things the alias cannot carry, because they are bodies not types:

- a step written against a span (`def has_raw_bytes(value: Raw[bytes]) -> bool:
  return value[1] >= 0`) must be rewritten to compare the value
  (`value == b"priority"`). There are **4** such bodies, all in
  `tests/run/test_compiled_fallback.py`.
- a step comparing against a `raw_str()` code (`value == PRIORITY`) becomes
  `value == "priority"`. There are **2**.

**Tests to drop, with the reason.** All of `tests/run/test_rows.py` (4 tests) —
they pin an annotation the design removes; the behaviour they cover (a list
column summed per row, a row with no items, several fields) must be re-pinned
against `list[Item]`, which is a better test because it also covers `str` fields.
In `test_compiled_fallback.py`: `test_raw_annotations_use_internal_representations`,
`test_raw_string_gives_the_same_answer_in_every_mode`,
`test_raw_string_constants_are_preconverted`, and the `Raw[str]` half of
`test_str_output_falls_back_but_raw_str_output_joins_the_kernel`. Keep the
semantic-`str` and semantic-`bytes` tests unchanged — those pin the surface this
note recommends.

**What a user who already wrote plain types loses: nothing.** The recommendation
makes their existing code the documented path.

---

## 7. Ranked recommendation

**Ship now**

1. **Make the boundary promises 1-4 of §5 true.** Reject a fractional float into
   an `int`; apply the null rules inside a `TypedDict` field; name the step, the
   input, the index and the field in every boundary error. This is the whole
   user-visible value in this note, it is independent of the annotation
   question, and it is what the 24 real projects actually got hurt by.
2. **Delete `Raw`, `Rows` and `raw_str` from `__all__`**, behind one release of
   `DeprecationWarning` aliases. Zero users, ten measured hazards, negative
   measured benefit for `Rows`.
3. **Document `list[Item]` / `Item` (a `TypedDict`) as the nested surface**, and
   say plainly that a `TypedDict` is what gets JSON fields coerced while a bare
   `dict` is the opt-out. Four projects reinvented this; it should not be folklore.
4. **Delete the struct-of-arrays workaround** from the example projects and say
   in the guide that a step may return `list[dict]`. It works now (§4) and the
   workaround is copy-pasted through four of them.
5. **Fix H12** (numba's `AssertionError` escaping the fallback) — one line in
   `FALLBACK_ERRORS`. It is an empty-message crash today.

**Leave**

6. `Table` / `param_table` exactly as they are: documented, working, and a
   *params* construct, not request data.
7. `str` staying outside the fused kernel. 5 µs on a single record is not worth a
   surface. If the 4.1M-row batch case (F5: 294 rows/s, 3.9 h against a 3 h
   window) needs the 4.7x, make it an **engine** decision keyed on `n`, inside
   `njit.py`, invisible to the user.
8. `bytes` as a semantic type only. Do not expose the span. Delete `-> bytes`
   from the kernel path or refuse it like `-> str` (H11).

**Do not do**

9. Do not add `Struct[...]`, `Rows[...]`, or any third spelling for nested data.
10. Do not extend `Raw` to nested types. It would multiply the fill/null holes of
    H2/H3/H5/H6 across four more types.

### The biggest risk of getting this wrong

The risk is not slowness; it is a wrong number that nobody sees. Every
representation annotation in this area already has a path where the value's type
changes behind the user's back while their source stays the same: a `Raw[str]`
step that stops compiling because someone added a helper call receives a `str`
instead of a code and answers `False` instead of `True` (H1), and a `Rows[Item]`
step turns a JSON `null` into `NaN` where the plain type raised (H7). Both live
in credit and pricing paths, both survive every mode, and only
`assert_equivalent` catches the first. If the surface keeps letting users name a
representation, decider is asking every author to hold in their head a second,
invisible type system whose bindings shift with unrelated edits to their own
code — and the failure mode is a silently approved loan, not an exception. The
cost of over-correcting is a batch job that takes 4x longer, which is visible on
a dashboard the same afternoon. Those two risks are not comparable, and the
design should not trade the first for the second.
