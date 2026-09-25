# decider: gaps found building a combinatorial-search pipeline

The document below was agent feedback ive added my comments in >>><<< as i reviewed the doc to add my own opinion as well but mostly i agree with the items in this document.

Found while building a real pipeline on decider (commit `58634da`, 2026-09-25). The pipeline has the same shape
as this running example:

> An **order** arrives with a list of **line items**. The pipeline tries every **bundle** (subset) of items. For
> each bundle it prices a shipping offer by looping over a **rate card** (a `param_table`), scores the bundle, and
> keeps the best one. That's up to 2^n − 1 bundles per order, each with an inner loop over rate-card tiers.

The pipeline was built twice. The first version uses decider constructs only (`decider.loop` over bundles, a
nested `decider.loop` over tiers). The second calls the same step functions from one hand-written numba search
kernel. On a 9-item order (511 bundles) the decider-native version took 119 ms and the compiled search 2.4 ms. The
compiled version was also faster than a hand-written standalone numba kernel from 12 items up.

The native version was slow for one reason: a single step inside the bundle loop reads the order's item list, so
it runs in Python and stops the loop from fusing. Swap that step for one that compiles, and decider packs the
whole bundle loop (the tier loop and a branch inside it included) into one kernel. The native pipeline then
takes 1.4 ms at 9 items and 19 ms at 16 items (65,535 bundles), as fast as the hand-written search. Gap 3 is
therefore the one that matters most.

Each gap has **today** (code that shows the problem, runnable against that commit where possible) and **ideal**
(a sketch of an API that would fix it). The sketches are proposals, not designs.

Contents:
1. [Silent Python fallback for list inputs](#1-silent-python-fallback-for-list-inputs)
2. [Silent Python fallback when a step calls a plain helper](#2-silent-python-fallback-when-a-step-calls-a-plain-helper)
3. [Lists of records have no grain of their own](#3-lists-of-records-have-no-grain-of-their-own)
4. [No fan-out and reduce (search over generated candidates)](#4-no-fan-out-and-reduce)
5. [Withdrawn: calling a flow from a hand-written kernel](#5-withdrawn-calling-a-flow-from-a-hand-written-kernel)
6. [A branch condition must be a function step](#6-a-branch-condition-must-be-a-function-step)
7. [Stale disk cache when a callee changes](#7-stale-disk-cache-when-a-callee-changes)
8. [Functions with function arguments can't be disk-cached](#8-functions-with-function-arguments-cant-be-disk-cached)
9. [Frame step output dtypes are lost between frame steps](#9-frame-step-output-dtypes-are-lost-between-frame-steps)
10. [Typo heuristic flags legitimate input names](#10-typo-heuristic-flags-legitimate-input-names)
11. [Per-record overhead](#11-per-record-overhead)
12. [Template nests the package](#12-template-nests-the-package)
13. [Design note: should helper calls be traced?](#13-design-note-should-helper-calls-be-traced)

---

## 1. Silent Python fallback for list inputs

A step that reads a `list` input can't be compiled, so it runs as plain Python. That's reasonable, but it happens
with no warning, and `Engine(strict_compile=True)` doesn't raise either. The docs say fallbacks warn and strict
mode raises.

**Today**

```python
from typing import TypedDict
from decider import Engine, flow
from decider.engine.compile import Fallback   # private

class Item(TypedDict):
    price: float
    qty: int

def order_total(items: list[Item]) -> float:
    return sum(i["price"] * i["qty"] for i in items)

exe = Engine(strict_compile=True).bind(flow(order_total, name="order"), mode="fused")
exe.score({"items": [{"price": 2.0, "qty": 3}]})     # {..., 'order_total': 6.0}: no warning, no error

# The only way to find out, through private attributes:
[u.reason for u in exe.runner.units.values() if isinstance(u, Fallback)]
# ["reads 'items' as list[Item], which no kernel takes"]
```

`compile_call` already produces the reason; nothing reports it.

**Ideal**

```python
Engine(strict_compile=True).bind(pipeline, mode="fused")
# CompileError: order/order_total reads 'items' as list[Item], which no kernel takes; ...

exe = Engine().bind(pipeline, mode="fused")
# UserWarning: order/order_total runs in Python, row by row: reads 'items' as list[Item], ...

exe.fallbacks()      # public: {"order/order_total": "reads 'items' as list[Item], which no kernel takes"}
```

A public `fallbacks()` also lets a project write the test it actually wants: "only these steps run in Python".

---

## 2. Silent Python fallback when a step calls a plain helper

A step compiles, but a plain function it calls doesn't: numba raises
`TypingError: Untyped global name 'helper'`, and decider quietly runs the whole step in Python. Again there's no
warning and strict mode doesn't raise. A missing `@njit` on a helper turns a hot step into Python without anyone
noticing.

**Today**

```python
from decider import Engine, flow

def discounted(price):                  # a plain helper
    return round(price * 0.9, 2)

def net_price(price: float) -> float:
    return discounted(price)

exe = Engine(strict_compile=True).bind(flow(net_price, name="p"), mode="fused")
exe.score({"price": 10.0})              # {'price': 10.0, 'net_price': 9.0}, silently in Python

# The workaround today: decorate every helper
import numba

@numba.njit(cache=True)
def discounted(price):
    return round(price * 0.9, 2)
```

**Ideal**

decider already walks a step's globals to fingerprint it, so it can see the helpers a step reaches.

```python
exe = Engine().bind(flow(net_price, name="p"), mode="fused")
# either: decider jits plain helpers it reaches (with the same CPython round/pow overloads), or
# UserWarning: p/net_price calls 'discounted', which isn't compiled; decorate it with @numba.njit
```

See also gap 7: once helpers are jitted, their changes must also invalidate the step's disk cache.


>>> User Feedback

```python
def discounted(price):
    return round(price * 0.9, 2)

def net_price(price: float) -> float:
    return discounted(price)
```
hides the ability for us to see that the step net_price makes a call to the helper discounted which i feel hides the tracing a bit. Though i do see the need for helper functions as we dont always want to split this into 2 step or 3 steps for no reason. Im thinking something like
```python
@helper
def discounted(price):
    return round(price * 0.9, 2)

def net_price(price: float) -> float:
    return discounted(price)
```

and on the default example
```python
exe = Engine(strict_compile=True).bind(flow(net_price, name="p"), mode="fused")
exe.score({"price": 10.0}) 
```
maybe needs a warning or an error or something like
```
Step x made use of helper funciton 'discounted' and could not be run in optimal mode consider wrapping discounted with @helper to get better performace
```

we could dig in and try to automatically wrap all funciton calls the problem is if internally we make use of something like
```python
from requests import post

def call_out_external_service(value):
    return post(value)
```
then its okay for that function to not be jitted and we shouldnt be forcing the behaviour. maybe a warning to say that call_out_external_service does nto follow best practices and may result in suboptimal execution but i dont think it should be a forcing matter.
<<<


---

## 3. Lists of records have no grain of their own

The biggest gap. A request often carries a list of records (line items, accounts, parcels). Two kinds of logic
need them:

- **Per-item rules.** "An item is heavy if its weight is over the heavy limit; a missing weight means 0." These are
  ordinary scalar rules, and should be ordinary steps with `missing_as`, params and tracing.
- **Parent logic over its items.** "The total price of the items in bundle `mask`." This runs thousands of times
  per order, inside a loop, so it needs the order's items as arrays in a kernel.

Today neither is possible with steps. A loop node doesn't help: it gives iteration, not storage. On iteration `j`,
a step still has to read item `j` from somewhere on the row, and a row only holds scalars, or a list that forces
Python. A separate item frame (one row per item) does let per-item rules be normal steps, but no construct links
the two grains inside a pipeline.

**Today**

Per-item rules end up as polars expressions in a frame step. That re-implements `missing_as` by hand, breaks when
no item sends a key, and needs an explicit schema, because an all-empty list arrives as `List(Null)`:

```python
import polars as pl
from decider import frame_step, param

ITEM_SCHEMA = pl.List(pl.Struct({"price": pl.Float64, "heavy": pl.Boolean}))

@frame_step(reads=["items"], writes=["clean_items"])
def clean_items(df: pl.DataFrame, heavy_kg: float = param(20.0)) -> pl.DataFrame:
    inner = df.schema["items"].inner
    sent = {f.name for f in inner.fields} if isinstance(inner, pl.Struct) else set()

    def field(name, dtype, fill):              # missing_as, re-implemented per field
        value = pl.element().struct.field(name).cast(dtype) if name in sent else pl.lit(None, dtype)
        return value.fill_null(fill)

    return df.with_columns(clean_items=pl.col("items").list.eval(pl.struct(
        price=field("price", pl.Float64, 0.0),
        heavy=field("weight", pl.Float64, 0.0) > heavy_kg,
    )).cast(ITEM_SCHEMA))
```

Parent logic over items is either a step reading the list (Python, per gap 1), or an `@njit` helper fed arrays
that are rebuilt from the list on every call:

```python
@numba.njit(cache=True)
def bundle_price(mask, price):                 # untraced helper
    total = 0.0
    for j in range(len(price)):
        if (mask >> j) & 1:
            total += price[j]
    return total

def bundle_total(mask: int, clean_items: list[dict]) -> float:     # Python: it reads a list
    return bundle_price(mask, np.array([i["price"] for i in clean_items]))
```

Inside a `decider.loop` over bundles, that Python step dominates: about 0.2 ms per iteration.

**Ideal: child frames**

A request can carry child rows. Steps written for the child grain run on them (compiled, traced, with `missing_as`
and params). A parent-grain step reads its own children as a namedtuple of arrays, the same representation a
`param_table` already arrives in.

```python
from decider import Rows, each, flow, missing_as, param, step

# Item grain: ordinary steps
def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg

item = flow(heavy, name="item")

# Order grain: reads the order's items as arrays, in a kernel
@step(output="bundle_total")
def bundle_total(mask: int, items: Rows) -> float:
    total = 0.0
    for j in range(len(items.price)):
        if (mask >> j) & 1:
            total += items.price[j]
    return total

pipeline = flow(
    each("items", item, name="items"),     # runs `item` on one row per list element; writes the list back
    bundle_total,                          # compiled, and fusable into a surrounding loop
    name="order",
)
pipeline.run(orders)                                  # items as a list column (JSON requests work unchanged)
pipeline.run(orders, children={"items": items_df})    # or as a separate frame, linked by a key
```

What it gives:

- per-item rules are steps: in the graph, in sessions (`break_at("order/items/heavy")` shows every item's value),
  params under `order/items/...`;
- a single JSON request needs no handler changes: the list is the child frame;
- a loop whose body reads `items: Rows` can pack into one kernel.

Why this matters most: it's what decides whether native decider is fast. In the project, the bundle loop
(`decider.loop` over bundles, a nested `decider.loop` over rate-card tiers, a `branch` inside) was driven from
Python, only because the step that sums the bundle's items reads a list. With that one step replaced by a
compilable stand-in, the whole loop packed into a single kernel:

| items | bundles | native loop today | native loop, list step compilable | hand-written numba search |
|---:|---:|---:|---:|---:|
| 9 | 511 | 119 ms | 1.4 ms | 2.4 ms |
| 16 | 65,535 | – | 19 ms | 61 ms |

(The stand-in does a little less work per bundle than the real step, so read these as indicative.) With list
inputs in kernels, the native pipeline is the fast pipeline, and gap 5 isn't needed at all.

Implementation hint: Arrow lists are already offsets plus child buffers. The child grain is the child buffers run
through the existing runner. `Rows` for parent row `r` is each child column sliced `offsets[r]:offsets[r + 1]`,
passed as the same namedtuple type `param_table` uses, so the kernel side should mostly exist already.


>>> User Feedback
I like the above suggestion i also just want to ensure we can handle lists elegantly in decider the user should be able to write something like:
```python
class Item(TypedDict): # Or whatever is appropriate even if its a base class we provide
    el_1: int
    el_2: str

def find_best_item(items: List[Item]):
    for i in items:
        if el_1 == 400 and el_2 == "snoop":
            return i
    return None # or even methods that return lists 
```
and even if its suboptimal it should run with no issues. the user should be able to do something though like
```python
class Item(Struct): # Or whatever base class is appropriate
    el_1: int
    el_2: str

def find_best_item(items: decider.List[Item]):
    for i in items:
        if i.el_1 == 400 and i.el_2 == "snoop":
            return i
    return None # or even methods that return lists 
```
and then it should be able to make very optimal code. ideally they user shouldnt even have to do that but im not sure what the limitations of numba is. 
<<<
---

## 4. No fan-out and reduce

"Generate N candidates per request, score each with a flow, keep the best per request" is a common decision shape:
best bundle, best term, best price point, best allocation. decider has no construct for it:

- a frame step can't change the row count, so candidates can't become rows;
- a `decider.loop` over candidates works, but its body is driven from Python per iteration unless every step in it
  compiles (and with gap 3, the steps reading items don't).

**Today**

```python
import polars as pl
from decider import frame_step

@frame_step(reads=["order_id"], writes=["bundle"])
def bundles(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(pl.DataFrame({"bundle": range(1, 8)}), how="cross")
# ValueError: frame step order/bundles returned 21 rows for 3
```

So the search went into a hand-written, generic numba harness, called from a frame step. This is the harness
(domain-free; it could be a starting point):

```python
from collections import namedtuple
import numpy as np
from numba import njit

Result = namedtuple("Result", ["best", "score", "evaluated", "disqualified"])

def search(count, disqualify, evaluate, requests, items, offsets, policy, width):
    """count/disqualify/evaluate are njit functions; request r owns items offsets[r]:offsets[r + 1]."""
    n = len(offsets) - 1
    result = Result(np.full(n, -1, np.int64), np.full((n, width), np.nan), np.zeros(n, np.int64), np.zeros(n, np.int64))
    _search(count, disqualify, evaluate, requests, items, offsets, policy, *result)
    return result

@njit(nogil=True)
def _search(count, disqualify, evaluate, requests, items, offsets, policy, best, score, evaluated, disqualified):
    for r in range(len(offsets) - 1):
        lo, hi = offsets[r], offsets[r + 1]
        top = score[r]
        for k in range(count(requests, r, items, lo, hi, policy)):
            if disqualify(requests, r, items, lo, hi, k, policy) != 0:      # reason code, 0 = keep
                disqualified[r] += 1
                continue
            s = evaluate(requests, r, items, lo, hi, k, policy)              # tuple of floats
            evaluated[r] += 1
            if best[r] < 0 or _better(s, top):                               # lexicographic, first wins ties
                best[r] = k
                for i in range(len(s)):
                    top[i] = s[i]

@njit(nogil=True)
def _better(s, top):
    for i in range(len(s)):
        if s[i] != top[i]:
            return s[i] > top[i]
    return False
```

Why `_better` exists: the rule is just Python's tuple comparison (`s > top` compares left to right, and
strictly-greater keeps the earlier candidate on a tie). numba supports `>` on tuples, but the harness can't keep
the best score as a tuple: numba needs every variable typed before it's read, and there's no score tuple until
the first candidate is evaluated (it may even be disqualified). So the best-so-far lives in a pre-allocated
array row, and comparing a tuple with an array row has to be spelled out:

```python
top = ...                                   # no value to start from: numba needs its type now
for k in range(n):
    s = evaluate(...)
    if best < 0 or s > top:                 # NotDefinedError: Variable 'top' is not defined
        best, top = k, s
```

A decider-owned `optimise` (below) knows the score's width and types from `rank=` at build time, so it can
generate `s > top` on a typed tuple (or an unrolled comparison) and drop the helper.

It's fast, but the frame step has to:
- redeclare every param `evaluate` needs;
- rebuild the request columns with the right null fills;
- flatten the item lists;
- call the step functions by hand, re-stating any loop they sit in.

None of that is traced.

Once gap 3 lands, a plain `decider.loop` over candidates already fuses, so `optimise` stops being needed for
speed. What it would still add is a declarative shape: candidates, named disqualify rules, a rank, and audit counts,
in place of hand-written carries (`best_*`, `keep_best`, `next_candidate`) and a re-evaluation of the winner.

**Ideal**

A row-preserving `optimise` step: one row in, the winner out.

```python
from decider import flow, optimise, param, subsets

def too_heavy(bundle_weight: float, max_weight: float = param(30.0)) -> bool:
    return bundle_weight > max_weight

best = optimise(
    candidates=subsets("items", limit=param(511)),      # or ranges, or a generator step; each has an id
    disqualify={"too_heavy": too_heavy},                 # named rules, checked before evaluation
    evaluate=flow(bundle_weight, bundle_total, shipping_offer, margin, name="bundle"),
    rank=["margin", "bundle_total"],                     # higher wins, left to right; a tie keeps the earlier one
    name="best_bundle",
)
pipeline = flow(each("items", item, name="items"), best, name="order")

out = pipeline.run(orders)
# per order: best_bundle.candidate, the winner's evaluate outputs, best_bundle.evaluated,
# best_bundle.disqualified (counts per reason); params nest under order/best_bundle/bundle/...
s = pipeline.session(orders)
s.break_at("order/best_bundle/bundle/margin")            # step through candidates like loop iterations
```

Compiled, `optimise` would generate one kernel per (candidates, disqualify, evaluate, rank), which avoids gap 8.
Interpreted, it runs as a loop over candidates, so modes stay equivalent.

---

## 5. Withdrawn: calling a flow from a hand-written kernel

An earlier draft asked for `Engine().kernel(flow)`: a decider flow compiled into a function that a hand-written
`@njit` kernel could call. The need was real in the project. Its search loop was hand-written numba, so the
per-candidate logic (a flow with its own loop over rate-card tiers) had to be copied into that kernel by hand:

```python
@njit
def search(...):
    for k in range(n_candidates):
        ...
        tier, rate, found = 0, 0.0, False            # a hand-written copy of the `shipping` flow's loop
        while _more_tiers(tier, found, rates):
            rate, found = _try_tier(tier, weight, rates)
            tier = _next_tier(tier)
```

The better principle: **decider owns orchestration, and user code never calls flows from inside its own kernels.**
Packaging a subgraph as a function called from inside another, user-written loop hides it from the graph and
from sessions, and invites tangled code. The need disappears once decider owns the loop:

- with gap 3, a `decider.loop` over candidates fuses into one kernel, so the search doesn't need hand-written
  numba in the first place (see the timings in gap 3);
- with gap 4, `optimise(evaluate=<flow>)` compiles the subgraph into decider's own generated kernel, as the fused
  runner already does for packed loops.

If decider needs "a flow compiled as a callable unit", that's an internal building block for those two, not a
public API.

---

## 6. A branch condition must be a function step

Branching on a bool column that already exists needs an identity step.

**Today**

```python
from decider import branch

branch("on_card", priced, skipped, modifies=["tier"], name="if_priced")
# TypeError: 'on_card' is not a step or a function

def on_card_now(on_card: bool) -> bool:     # identity step, only to branch
    return on_card

branch(on_card_now, priced, skipped, modifies=["tier"], name="if_priced")
```

**Ideal**

```python
branch("on_card", priced, skipped, modifies=["tier"], name="if_priced")      # a column name is a condition
```

---

## 7. Stale disk cache when a callee changes

numba keys a disk-cached kernel by its own function's bytecode and source file. It doesn't notice when a function
it calls changes (numba documents this limitation). `_SaltedCache` adds only the hash of `cpython.py`, so decider
steps inherit the problem: a step that calls an `@njit` helper in another module keeps serving the old machine
code after the helper changes. It happened in the project: an unchanged search kernel served a specialisation
compiled against an old evaluate function and failed with
`AttributeError: module ... has no attribute '<a namedtuple since removed>'`. The minimal example below
shows the worse case: a changed number and a silently wrong answer.

**Today**

```python
# rates.py
from numba import njit

@njit(cache=True)
def rate(amount):
    return amount * 0.10

# steps.py
from rates import rate

def fee(amount: float) -> float:
    return rate(amount)

# run.py
from decider import Engine, flow
from steps import fee
print(Engine().bind(flow(fee, name="p"), mode="fused").score({"amount": 100.0})["fee"])

# $ python run.py        -> 10.0   (disk cache written)
# edit rates.py: 0.10 -> 0.20      (steps.py unchanged)
# $ python run.py        -> 10.0   silently wrong: should be 20.0; fee's cached kernel is keyed by steps.py only
```

**Ideal**

decider already computes a content fingerprint that covers everything a function reaches (its globals, recursively,
functions by content). Put it in the cache key:

```python
class _SaltedCache(FunctionCache):
    def _index_key(self, sig, codegen):
        return (*super()._index_key(sig, codegen), SALT, fingerprint(self._py_func))
```

The project used exactly this for its own kernels, and the stale reads stopped.


>>>
maybe the @helper suggestion already fixes this as the main issue here is that the user should never have to write @njit over their own funcitons in my opinion unless its a really obscure case. we should eb the one that does the jitting scilently without the user knowing numba even exists. and for most cases the user wouldnt change the code when they deployed and they would work in interpreted mode while developing. so this should be a non-issue in most cases. ideally a `decider build` or something like that should be the only time the cache comes into play. but its worth reevaluating because it would be confusing if something was jitted and you changed it and no update happened when you tried debugging the code.
<<<
---

## 8. Functions with function arguments can't be disk-cached

Observed, **not reduced to a minimal repro**. Simple cases (a cached `apply(fn, x)` called from several scripts, a
pytest run then a script, a function from a module loaded under a name that can't be imported) all cached fine.

What happened: a generic kernel taking `@njit` functions as arguments (gap 4's `_search`, with namedtuple-of-array
arguments and decider's `param_table` bundles inside them) was disk-cached. A pytest run
(`--import-mode=importlib`) wrote its cache index, including specialisations whose function arguments were
defined in test modules. A later plain script compiled a new specialisation, and saving the index failed:

```text
File ".../numba/core/caching.py", line 636, in _dump
File ".../numba/core/types/functions.py", line 487, in __getnewargs__
    raise ReferenceError("underlying object has vanished")
ReferenceError: underlying object has vanished
when serializing dict item '_wr'
when serializing numba.core.types.functions.Dispatcher reconstructor arguments
```

The index keys hold `Dispatcher` types, and so weakrefs to dispatchers. Once a key's dispatcher is gone,
re-pickling the index fails. The project now compiles such kernels per process (about 3 s at cold start for
the whole pipeline) and disk-caches only the functions they call.

**Ideal**

If decider builds gap 4, it should generate a specialised kernel source per combination of functions, written to a
real file (as decider already does for generated code), so each is an ordinary, cacheable, content-keyed function
with no function-typed arguments.
>>> 
I think this again is just bad code where the user took control of njit themselves and i think it will be resolved as a biproduct of other items
<<<
---

## 9. Frame step output dtypes are lost between frame steps

A frame step writes a column with a precise dtype; the next frame step can get it back re-inferred. An all-empty
`List(Struct)` column comes back as `List(Null)`, so `struct.field(...)` fails downstream.

**Today**

```python
import polars as pl
from decider import Engine, flow, frame_step

SCHEMA = pl.List(pl.Struct({"price": pl.Float64}))

@frame_step(reads=["items"], writes=["clean"])
def clean(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(clean=pl.col("items").list.eval(pl.struct(price=pl.lit(0.0))).cast(SCHEMA))

@frame_step(reads=["clean"], writes=["seen"])
def seen(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(seen=pl.lit(str(df.schema["clean"])))

exe = Engine().bind(flow(clean, seen, name="p"), mode="fused")
exe.score({"items": []})["seen"]                              # 'List(Null)'
exe.run(pl.DataFrame({"items": [[]]}))["seen"][0]             # 'List(Null)'
exe.score({"items": [{"price": 1.0}]})["seen"]                # 'List(Struct({'price': Float64}))'
```

**Ideal**

Keep the polars Series a frame step returns (or its dtype), not values re-inferred from Python objects. Or let a
frame step declare it: `frame_step(writes={"clean": SCHEMA})`.

>>>
I agree with this ideally framesteps should show their output schema. we can add a AUTO type or something if it cant be given and has to be infered but by default i think its better that the schema is explicit here. 
<<<

---

## 10. Typo heuristic flags legitimate input names

Reading an unknown name as an input column warns when it looks like a name produced earlier (difflib, cutoff 0.8).
Real request fields hit this all the time, and the only fix is renaming produced values.

**Today**

```python
from decider import flow, missing_as, step

@step(outputs=("quote_handling_fee_pct", "quote_fee_cap"))
def rate_row(tier: int) -> tuple[float, float]:
    return 1.0, 2.0

def fee_cap_used(quote_fee_cap: float) -> float:
    return quote_fee_cap

def handling_fee(handling_fee_pct: float = missing_as(0.0)) -> float:    # a request field
    return handling_fee_pct

Engine().bind(flow(rate_row, fee_cap_used, handling_fee, name="p"), mode="interpreted")
# UserWarning: p/handling_fee: reading 'handling_fee_pct' as an input column. It looks like
# 'quote_handling_fee_pct' (produced by 'p/rate_row'); if it's a typo, rename it or relabel(...)
```

Pipelines with many similar money fields got several of these, each resolved by renaming.

**Ideal**

Any of:

```python
flow(rate_row, fee_cap_used, handling_fee, inputs=["handling_fee_pct"], name="p")   # declared request fields
# or: no warning for names present in sample_request.json at build time
# or: warn only when the name is missing from the frame at run time
```
>>>
im tempted to take this one step futher in the sense that maybe we should have a way to allow the user to add an input schema (that we can cast and validate) or even just the input fields that way its easier to validate if somethign is missing. like if an input schema is provided and the flow needs a variable not in inputs we can error before a payload is ever given.
<<<
---

## 11. Per-record overhead

On the single-record path (`exe.score(dict)`), decider costs roughly 30–40 µs per step. A ~50-step pipeline takes
about 1.5–2 ms per record even when every kernel is trivial. Frame steps add polars overhead (each `select` is tens
of µs). For comparison, the same work as one hand-written kernel takes about 0.06 ms.

**Today**

```python
import time
exe = Engine().bind(pipeline, mode="fused")      # ~50 steps: 3 packed loops, 2 frame steps
exe.score(record, params)
t = time.perf_counter()
exe.score(record, params)
(time.perf_counter() - t) * 1000                 # ~2 ms
```

**Ideal**

A single-record fast path. When a plan's scalar sections have no frame steps between them, compile the whole
section, glue included, into one kernel that takes the record's values directly: a scalar-argument `fused`, not
arrays of length 1. Cache converted params bundles per call site (partly done already).

>>>
im debating this one. i think the overhead is worthwhile that we treat the polars path and the python path identically and always make use of that shim. at the end of the day 2ms isnt that much for 50 steps. That being said maybe its worth a consideration on the backlog we could have ultra low latency cases where it does matter.
<<<

---

## 12. Template nests the package

`decider template NAME` writes `NAME/NAME/pipeline.py`. A project directory that is itself the package reads better
and avoids shadowing just as well:

```text
order_pricing/            # the project directory is the package
  __init__.py
  pipeline.py             # DECIDER_API__PIPELINE=order_pricing.pipeline:build
  inference.py
  lib/                    # the steps
  configs/0.0.0/params.json
  sample_request.json
  tests/
  .env
```

Imports stay qualified (`order_pricing.pipeline`). The repository root, not the project directory, goes on
`sys.path` (one editable install covering several projects with `dev-mode-dirs = ["."]`). Two projects in one
repository never shadow each other, and one can import the other's steps.

>>>
I think we also need an explicit path on where to add extensions. In the old decider we had decider_extensions. I think here we should have ext/ where the user can add externally developed modules to be used in the flow. and we can have a similar initialise_extensions like we had iniitalise_decider in v1 of decider (what is in the main branch)
<<<

---

## 13. Design note: should helper calls be traced?

Helpers called from steps don't show up in the graph, in `session.value()` or in outputs. A "traced call"
primitive was considered and **not** recommended:

- a helper inside a hot loop runs thousands of times per record; recording each call costs more than the work;
- the Python debugger integration ("step into the Python") already steps through helpers line by line, in
  interpreted mode.

The rule that worked instead: **anything with a business meaning is a step; helpers are only arithmetic inside one
step.** For that rule to be safe, decider needs gap 2 (never silently lose compilation because of a helper) and
gap 7 (a helper change must invalidate the cache). Most helpers disappear once decider has gap 3 (and
optionally gap 4), because the remaining reason for one is reaching data that steps can't.

>>>
again its a bit of a tradeoff if we are able to do it in a way that is low latency or basically no overhead when not in the interpreted runtime I think its worthwhile to have a way to be able to see these calls or at least in static analysis know what calls are made. it doesnt have to affect the fused mode at all where the performance matters the most.
<<<

---

## What worked well

- **One param object per policy value, used as the default wherever it's needed.** Steps, frame steps and two
  pipelines read one params document, and a test asserts both pipelines declare the same params. It's worth
  documenting as a pattern:

  ```python
  # policy.py
  HEAVY_KG = param(20.0, ge=0.0, shared_key="heavy_kg")

  def heavy(weight: float = missing_as(0.0), heavy_kg: float = HEAVY_KG) -> bool: ...

  @frame_step(reads=[...], writes=[...])
  def search(df: pl.DataFrame, heavy_kg: float = HEAVY_KG) -> pl.DataFrame: ...
  ```

- **`param_table`** for rate cards: read from steps inside packed loops, and from frame steps.
- **Nested loops with a branch inside pack into one kernel each** in fused mode.
- **Sessions pause inside loops:** `break_at("order/search/bundle/keep_best")` shows each candidate in turn.
- **`relabel` for placing one step twice** (e.g. `shipping_cost` on the lightest and on the heaviest parcel) kept the
  per-stage steps small.
