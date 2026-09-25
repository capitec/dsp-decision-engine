# TASK: the migration-search benchmark

A self-contained problem for improving decider. Build it on decider, measure it, change decider in general-purpose
ways, and measure again. The problem is designed so a natural implementation hits the hard parts of decider:
- lists of records inside a request;
- a search over thousands of generated candidates per request;
- a nested search over a parameter table;
- float arithmetic that has to match to the cent;
- a tie-broken multi-key ranking.

A pure-Python reference implementation and a seeded case generator are included below. They define the answer.

## Ground rules

- **General-purpose changes only.** Every change to decider must make sense for any user. Don't add anything that
  names or special-cases this benchmark (no `migration`, `subset`-for-this-problem, or `workload` concepts in
  decider). If a construct is added (say, a way to iterate over a request's list of records), it must be documented
  and tested as a general feature.
- **Public API only in the benchmark project.** If the project needs a private import, that's a decider gap to
  record and fix.
- **Keep decider's own test suite green** after every change, and add tests for what you add.
- **The reference is normative.** Where this text and `reference.py` disagree, `reference.py` wins. Don't change
  the reference or the generator; they are the fixed yardstick.

## The problem

A cloud tenant runs **workloads**, some on its own legacy hosts and some on third-party vendors. It asks for a
**reserved-capacity deal**: an amount of prepaid capacity, paid back monthly over a chosen **length** (months). The
deal may also fund migrating some of the tenant's workloads onto the new capacity. Migrating a workload costs a
one-off migration fee, which the deal must cover. The workload's monthly spend then stops counting against the
tenant's budget, and any migration waives the deal's setup fee.

For one request the decision is:

1. **Try every subset of workloads to migrate.** Subset `mask` migrates workload `j` when bit `j` of `mask` is set
   and the workload is movable. Masks run from 0 (migrate nothing) to `min(2**n - 1, max_subsets)`.
2. **Per subset, work out what the tenant is still committed to:** unmigrated vendor spend by class, overdue
   invoices weighted by class, legacy spend, and a budget left over (or a fixed override from the request).
3. **Per subset, find the longest deal length that works.** Walk the length ladder (a policy table, one row per
   length) from `max_length` down to `min_length`. Price each length:
   - the amount range (it must cover migration fees);
   - a capped and floored rate;
   - a setup fee (waived when anything migrates);
   - the monthly charge;
   - if the charge is above the cap, cap it and solve back for the amount it pays for.

   Take the first length whose largest amount fits the budget left over and the limits.
4. **Score the subset** as `(max_amount, headroom, monthly_saving)`, where headroom is the amount left after
   migration fees. A subset with no deal, or no headroom, scores zeros.
5. **Keep the best subset:** higher scores win, compared left to right; a tie keeps the lower mask.
6. **For the winner, also find the requested fit:** the length nearest the tenant's `requested_length` whose amount
   range holds `requested_amount`. Search upward if the requested length is below the shortest length on the
   ladder, downward otherwise. If none fits, fall back to the largest amount within the exposure limit (a tie keeps
   the longer length).
7. **Answer with every intermediate value of the winner,** plus the ids of the migrated workloads.

### Request (one JSON object per decision)

| field | type | missing means |
|---|---|---|
| `tenant_id` | str | |
| `reserved_requested`, `spot_requested` | bool | which ladder: reserved if requested, else spot if requested, else none (no deal) |
| `monthly_budget` | float | required |
| `budget_override` | float or null | if present, it *is* the budget left over |
| `billed_spend`, `other_commitments`, `general_allowance`, `cache_allowance` | float | 0 |
| `billing` | `"monthly"`, `"weekly"`, `"fortnightly"` or null | monthly; weekly/fortnightly rebase vendor spend by `12 / billing_periods_per_year` |
| `billing_periods_per_year` | int | 12 |
| `max_length`, `min_length` | int | required; `max_length` 0 means no deal |
| `capacity_limit`, `exposure_limit` | float | required |
| `base_rate`, `support_fee` | float | required |
| `rate_adjustment`, `risk_margin`, `priority_score` | float | 0 |
| `requested_length`, `requested_amount` | int, float or null | no requested fit unless both are present |
| `workloads` | list of workload objects | empty |

Workload: `id`, `host` (`"legacy"`, `"vendor"` or anything else), `class_code` (`D`/`E` protected, `C` cache,
`1`–`4`/`B` general, anything else other), `family` (`"gpu"`, `"storage"`, other), `burst`, `migration_cost`,
`monthly_cost`, `units`, `quota`, `overdue_amount`, `overdue_age_days` (0 or missing: unknown, so the overdue
amount is ignored), `months_left`, `disputed`, `movable` (only an explicit `false` makes it unmovable). Any field may
be missing or null.

### Policy (the params document; the same for every request)

All scalar entries of `POLICY` in the reference are policy params. `reserved_ladder` and `spot_ladder` are policy
tables with columns `length` (int), `min_amount`, `max_amount`, `discount`, `quoted_rate`, `quoted_support`
(floats). Each generated case comes with its own policy overrides (`generate.cases()` yields `(request, policy)`).
Run each case with that case's params document, as decider runs a call with a params document.

### Answer

Every key `reference.decide()` returns: `best_subset`, `migrated_ids`, the 18 subset aggregates, the four
commitment values, the three score parts, `has_offer`, `has_fit`, `fit_exact`, and `offer_*`/`fit_*` for `length`,
`min_amount`, `max_amount`, `min_charge`, `max_charge`, `rate`, `surcharge`. Absent offers are zeros.

## What to build

1. **`native/`: a decider project, decider constructs only.** Use `decider template`. It takes the request JSON
   exactly as above and answers with exactly the reference's keys and values. `decider build` and `decider serve`
   must work, and `POST /invocations` with a batch (a JSON array) must work too. Policy lives in
   `configs/<version>/params.json`, with the ladders as table params. Aim for the code a careful decider user
   would write:
   - every rule with a business meaning is a step;
   - loops are decider loops;
   - nothing is hand-written numba.
2. **`baseline_numba.py`: the speed yardstick, no decider.** A hand-written numba implementation with the same
   answers, for timing comparisons only. Don't import it from `native/`.
3. **`bench.py`.** Runs the gates and prints the metrics below.

If a natural implementation of `native/` is impossible or slow because of decider, don't work around it in the
project. Record the gap, fix decider in a general way, and simplify the project.

## Gates (must pass before a metric counts)

- **Parity:** `generate.cases(seed=1, count=500)`, every answer key equal to `reference.decide(request, policy)`,
  compared with `==` (floats to the last bit, which the reference's operation order makes reproducible). Check it in
  every engine mode (`interpreted`, `stepped`, `fused`), on the single-record path (`score`) and on the batch path
  (`run` over all 500 requests at once, grouped by policy or one call per policy).
- **Serving:** `decider build`, then `decider serve`, then `curl` with `sample_request.json` returns the reference's
  answer.
- **No silent fallbacks:** state which steps run in Python and why; the target is none.

## Metrics (report all of them after every iteration)

Speed, fused mode, median of warm runs, uncapped subsets (`max_subsets = 2**n - 1`), a 7-row ladder
(`length` 3–60), `reserved_requested` true:

| metric | native | baseline_numba | reference (Python) |
|---|---:|---:|---:|
| 1 workload, single record | | | |
| 5 workloads | | | |
| 9 workloads (511 subsets) | | | |
| 12 workloads (4,095) | | | |
| 16 workloads (65,535) | | | |
| batch: 1,000 requests × 8 workloads, one `run` | | | |
| cold start: import to first answer, fresh process, warm disk cache | | | |

Shape of the native project:

| metric | target |
|---|---|
| steps that run in Python (with reasons) | 0 |
| frame steps (with reasons) | 0, or only where the logic truly spans records |
| hand-written numba lines | 0 |
| non-step helper functions (with reasons) | only pure arithmetic inside one step |
| orchestration written twice (a loop re-stated by hand anywhere) | none |
| private decider imports | none |
| policy values outside the params document | none |
| lines of Python (excluding tests) | report |

Debuggability (yes/no): a session can pause inside the subset loop and inside the length loop and show that
iteration's values. Item-level values (e.g. each workload's recent overdue amount) are visible in a session.

**Targets:**
- native fused within 1.5× of `baseline_numba` at 9 and 16 workloads;
- single-record overhead under 0.5 ms at 1 workload;
- every shape target met.

On the reference machine, `reference.py` takes about 8 ms at 9 workloads and about 330 ms at 14.

## Iterating

1. Build `native/` with decider as it is today, pass the gates, and record the baseline metrics.
2. Pick the biggest gap between the metrics and the targets. Find its cause in decider, and fix it generally
   (with tests and docs).
3. Simplify `native/` to use the fix, pass the gates again, and re-record the metrics.
4. Keep a log (`notes/benchmark-migration.md`): per iteration, the change, the metrics table, and what's still in
   the way.

Expect friction around:
- list-of-record inputs (nulls inside items, string codes, per-item rules);
- loops that can't fuse because one step can't compile;
- nested loops with branches;
- table params read inside loops;
- string comparisons in kernels;
- tuple-valued ranking;
- disk-cache invalidation when a function a step calls changes;
- single-record overhead.

These are hints, not prescriptions: measure first.

## `reference.py`

```python
"""Reference: the best set of workloads to migrate into one reserved-capacity deal."""

PROTECTED, CACHE, GENERAL, OTHER = 0, 1, 2, 3
CLASS_BUCKETS = {"D": PROTECTED, "E": PROTECTED, "C": CACHE, "1": GENERAL, "2": GENERAL, "3": GENERAL,
                 "4": GENERAL, "B": GENERAL}
REBASED_BILLING = ("weekly", "fortnightly")

POLICY = {
    "max_subsets": 511,
    "overdue_recent_days": 45,
    "protected_overdue_pct": 0.0, "general_overdue_pct": 0.0, "other_overdue_pct": 0.0,
    "disputed_overdue_pct": 0.0,
    "short_length": 6, "short_ceiling": 30.0, "long_ceiling": 26.0, "rate_floor": 9.0,
    "surcharge_cutoff": 650.0, "surcharge_per_unit": 0.8,
    "setup_flat": 150.0, "setup_pct": 2.5, "setup_cap": 900.0,
    "charge_cap": 5000.0, "minimum_amount": 0.0,
    "promo_threshold": 12345.0, "promo_override": 15000.0,
    "reserved_ladder": [], "spot_ladder": [],
}

AGGREGATES = (
    "legacy_migration_cost", "vendor_migration_cost", "total_migration_cost",
    "legacy_freed", "vendor_freed", "burst_freed",
    "general_spend", "cache_spend", "protected_overdue", "general_overdue", "other_overdue", "disputed_overdue",
    "legacy_spend", "legacy_units", "longest_months_left", "gpu_units", "gpu_quota", "storage_quota",
)
OFFER = ("length", "min_amount", "max_amount", "min_charge", "max_charge", "rate", "surcharge")


def _num(w, key):
    return float(w.get(key) or 0.0)


def normalise(workload, recent_days):
    age = workload.get("overdue_age_days")
    overdue = _num(workload, "overdue_amount")
    # An age of 0 or a missing age is unknown: the overdue amount is ignored.
    recent = bool(age) and age <= recent_days and overdue > 0
    return {
        "id": workload.get("id"),
        "legacy": workload.get("host") == "legacy",
        "vendor": workload.get("host") == "vendor",
        "movable": workload.get("movable") is not False,
        "bucket": CLASS_BUCKETS.get(workload.get("class_code"), OTHER),
        "gpu": workload.get("family") == "gpu",
        "storage": workload.get("family") == "storage",
        "burst": bool(workload.get("burst")),
        "migration_cost": _num(workload, "migration_cost"),
        "monthly_cost": _num(workload, "monthly_cost"),
        "units": _num(workload, "units"),
        "quota": _num(workload, "quota"),
        "recent_overdue": overdue if recent else 0.0,
        "disputed": bool(workload.get("disputed")),
        "months_left": int(workload.get("months_left") or 0),
    }


def aggregates(mask, items, plan_requested):
    a = dict.fromkeys(AGGREGATES, 0.0)
    a["longest_months_left"] = 0
    for j, it in enumerate(items):
        if (mask >> j) & 1 and it["movable"]:
            if it["legacy"]:
                a["legacy_migration_cost"] += it["migration_cost"]
            else:
                a["vendor_migration_cost"] += it["migration_cost"]
            if not plan_requested:
                continue
            if it["legacy"]:
                a["legacy_freed"] += it["monthly_cost"]
                if it["gpu"] or it["burst"]:
                    a["burst_freed"] += it["monthly_cost"]
            elif it["vendor"]:
                a["vendor_freed"] += it["monthly_cost"]
        elif it["vendor"]:
            if it["bucket"] == PROTECTED:
                a["protected_overdue"] += it["recent_overdue"]
                continue
            if it["bucket"] == CACHE:
                a["cache_spend"] += it["monthly_cost"]
            elif it["bucket"] == GENERAL:
                a["general_spend"] += it["monthly_cost"]
                a["general_overdue"] += it["recent_overdue"]
            else:
                a["other_overdue"] += it["recent_overdue"]
            if it["disputed"]:
                a["disputed_overdue"] += it["recent_overdue"]
        elif it["legacy"]:
            a["legacy_spend"] += it["monthly_cost"]
            a["legacy_units"] += it["units"]
            a["longest_months_left"] = max(a["longest_months_left"], it["months_left"])
            if it["gpu"]:
                a["gpu_units"] += it["units"]
                a["gpu_quota"] += it["quota"]
            elif it["storage"]:
                a["storage_quota"] += it["quota"]
    a["total_migration_cost"] = a["legacy_migration_cost"] + a["vendor_migration_cost"]
    return a


def commitments(a, req, p):
    overdue = (a["protected_overdue"] * (p["protected_overdue_pct"] / 100)
               + a["general_overdue"] * (p["general_overdue_pct"] / 100)
               + a["other_overdue"] * (p["other_overdue_pct"] / 100)
               + a["disputed_overdue"] * (p["disputed_overdue_pct"] / 100))
    general = max(a["general_spend"] - _num(req, "general_allowance"), 0.0)
    cache = max(a["cache_spend"] - _num(req, "cache_allowance"), 0.0)
    vendor = general + cache + overdue
    if (req.get("billing") or "monthly") in REBASED_BILLING:
        vendor = vendor * 12 / float(req.get("billing_periods_per_year") or 12)
    billed = max(_num(req, "billed_spend") - (a["vendor_freed"] + a["legacy_freed"]), 0.0)
    committed = max(billed, vendor) + _num(req, "other_commitments")
    committed_total = committed + a["legacy_spend"]
    override = req.get("budget_override")
    free_budget = float(override) if override is not None else req["monthly_budget"] - committed_total
    return {"vendor_commitment": vendor, "committed_spend": committed, "committed_total": committed_total,
            "free_budget": free_budget}


def surcharge_for(req, p):
    margin, score = _num(req, "risk_margin"), _num(req, "priority_score")
    if margin <= 0 or score < p["surcharge_cutoff"] or req["max_length"] <= 1:
        return 0.0
    return round(margin * p["surcharge_per_unit"], 2)


def price(row, migration_total, req, p):
    length = row["length"]
    max_amount = min(row["max_amount"], req["capacity_limit"])
    if migration_total > 0 and row["max_amount"] == p["promo_threshold"]:
        max_amount = p["promo_override"]
    min_amount = row["min_amount"] + migration_total
    surcharge = surcharge_for(req, p) if length > p["short_length"] else 0.0
    rate = round(req["base_rate"] + _num(req, "rate_adjustment") - row["discount"] + surcharge, 4)
    ceiling = p["long_ceiling"] if length > p["short_length"] else p["short_ceiling"]
    if row["quoted_rate"] > 0:
        ceiling = min(ceiling, row["quoted_rate"])
    rate = min(max(rate, p["rate_floor"]), ceiling)
    support = min(req["support_fee"], row["quoted_support"]) if row["quoted_support"] > 0 else req["support_fee"]
    waived = migration_total > 0
    monthly = (1 + rate / 100) ** (1 / 12) - 1
    growth = (1 + monthly) ** length
    factor = monthly * growth / (growth - 1)

    def setup(amount):
        return 0.0 if waived else min(p["setup_flat"] + amount * p["setup_pct"] / 100, p["setup_cap"])

    def charge(amount):
        return round((amount + setup(amount)) * factor + support, 2)

    min_charge, max_charge = charge(min_amount), charge(max_amount)
    if max_charge > p["charge_cap"]:
        max_charge = p["charge_cap"]
        principal = (p["charge_cap"] - support) / factor          # amount + setup fee
        amount = principal
        if not waived:
            amount = (principal - p["setup_flat"]) / (1 + p["setup_pct"] / 100)
            if p["setup_flat"] + amount * p["setup_pct"] / 100 > p["setup_cap"]:
                amount = principal - p["setup_cap"]
        max_amount = round(max(amount, 0.0), 2)
    return dict(zip(OFFER, (length, min_amount, max_amount, min_charge, max_charge, rate, surcharge)))


def ladder_for(req, p):
    if req.get("reserved_requested"):
        return p["reserved_ladder"]
    return p["spot_ladder"] if req.get("spot_requested") else []


def best_length(rows, migration_total, free_budget, req, p):
    if not rows or req["max_length"] <= 0:
        return None
    for length in range(req["max_length"], req["min_length"] - 1, -1):
        row = rows.get(length)
        if row is None:
            continue
        o = price(row, migration_total, req, p)
        if (o["max_amount"] < o["min_amount"] or o["max_charge"] > free_budget
                or o["max_amount"] > req["exposure_limit"] or o["max_amount"] < p["minimum_amount"]):
            continue
        return o
    return None


def requested_fit(rows, migration_total, req, p):
    wanted, amount = req.get("requested_length"), req.get("requested_amount")
    if not rows or wanted is None or amount is None:
        return None
    lengths = (range(wanted, req["max_length"] + 1) if wanted < min(rows)
               else range(wanted, req["min_length"] - 1, -1))
    for length in lengths:
        if length in rows:
            o = price(rows[length], migration_total, req, p)
            if o["min_amount"] <= amount <= o["max_amount"]:
                return {**o, "exact": True}
    best = None
    for length in range(req["max_length"], req["min_length"] - 1, -1):
        if length in rows:
            o = price(rows[length], migration_total, req, p)
            if o["max_amount"] <= req["exposure_limit"] and (best is None or o["max_amount"] > best["max_amount"]):
                best = o
    return None if best is None else {**best, "exact": False}


def score(offer, a, c):
    if offer is None:
        return 0.0, 0.0, 0.0
    headroom = offer["max_amount"] - a["total_migration_cost"]
    if headroom <= 0:
        return 0.0, 0.0, 0.0
    return offer["max_amount"], headroom, (a["legacy_spend"] + c["vendor_commitment"]) - offer["max_charge"]


def decide(req, policy=None):
    p = {**POLICY, **(policy or {})}
    items = [normalise(w, p["overdue_recent_days"]) for w in req.get("workloads") or []]
    rows = {row["length"]: row for row in ladder_for(req, p)}      # the last row wins for a repeated length
    plan = bool(req.get("reserved_requested") or req.get("spot_requested"))
    subsets = min(2 ** len(items) - 1, p["max_subsets"])
    best = None
    for mask in range(subsets + 1):                                 # mask 0 migrates nothing
        a = aggregates(mask, items, plan)
        c = commitments(a, req, p)
        offer = best_length(rows, a["total_migration_cost"], c["free_budget"], req, p)
        key = score(offer, a, c)
        if best is None or key > best[0]:                           # higher wins; a tie keeps the earlier mask
            best = (key, mask, a, c, offer)
    key, mask, a, c, offer = best
    fit = requested_fit(rows, a["total_migration_cost"], req, p)
    out = {"best_subset": mask,
           "migrated_ids": [it["id"] for j, it in enumerate(items) if (mask >> j) & 1 and it["movable"]],
           **a, **c,
           "max_amount_score": key[0], "headroom": key[1], "monthly_saving": key[2],
           "has_offer": offer is not None, "has_fit": fit is not None,
           "fit_exact": bool(fit and fit["exact"])}
    for prefix, o in (("offer", offer), ("fit", fit)):
        for field in OFFER:
            out[f"{prefix}_{field}"] = o[field] if o else 0
    return out
```

## `generate.py`

```python
"""Seeded requests and policies for the reference."""
import random

LENGTHS = [1, 3, 6, 9, 12, 18, 24, 36, 48, 60]


def ladder(rng):
    rows = []
    for length in sorted(rng.sample(LENGTHS, rng.randint(1, 7))):
        quoted = rng.random() < 0.3
        rows.append({
            "length": length,
            "min_amount": float(rng.choice([200, 500, 1000])),
            "max_amount": 12345.0 if rng.random() < 0.1 else float(length * rng.choice([400, 900, 1500])),
            "discount": rng.choice([0.0, 0.0, 1.25]),
            "quoted_rate": rng.choice([0.0, 14.0, 21.5]) if quoted else 0.0,
            "quoted_support": rng.choice([0.0, 20.0]) if quoted else 0.0,
        })
    return rows


def workload(rng, i):
    return {
        "id": f"w{i}",
        "host": rng.choices(["legacy", "vendor", "edge"], [0.45, 0.5, 0.05])[0],
        "class_code": rng.choice(["D", "E", "C", "1", "3", "B", "Z", None]),
        "family": rng.choice(["gpu", "storage", "web", None]),
        "burst": rng.random() < 0.1,
        "migration_cost": round(rng.uniform(0, 2500), 2),
        "monthly_cost": round(rng.uniform(50, 900), 2),
        "units": round(rng.uniform(0, 64), 1),
        "quota": round(rng.uniform(0, 128), 1),
        "overdue_amount": rng.choice([0.0, 0.0, None, round(rng.uniform(1, 2500), 2)]),
        "overdue_age_days": rng.choice([None, 0, 3, 30, 45, 46, 200]),
        "months_left": rng.choice([None, 0, 2, 11, 40]),
        "disputed": rng.random() < 0.15,
        "movable": rng.choice([True, True, True, True, False, None]),
    }


def policy(rng):
    return {
        "max_subsets": rng.choice([511, 511, 40, 3]),
        "protected_overdue_pct": rng.choice([0.0, 5.0]), "general_overdue_pct": rng.choice([0.0, 10.0]),
        "other_overdue_pct": rng.choice([0.0, 20.0]), "disputed_overdue_pct": rng.choice([0.0, 50.0]),
        "rate_floor": rng.choice([9.0, 16.0]), "minimum_amount": rng.choice([0.0, 800.0]),
        "charge_cap": float(rng.choice([400, 1500, 5000])),
        "reserved_ladder": ladder(rng), "spot_ladder": ladder(rng) if rng.random() < 0.7 else [],
    }


def request(rng, n=None):
    reserved, spot = rng.choice([(True, False), (True, False), (True, True), (False, True), (False, False)])
    req = {
        "tenant_id": f"t{rng.randrange(10**6)}",
        "reserved_requested": reserved, "spot_requested": spot,
        "monthly_budget": round(rng.uniform(1500, 9000), 2),
        "billed_spend": round(rng.uniform(0, 5000), 2),
        "other_commitments": rng.choice([0.0, 150.0, None]),
        "general_allowance": rng.choice([0.0, 200.0]), "cache_allowance": rng.choice([0.0, 80.0]),
        "billing": rng.choice(["monthly", "monthly", "weekly", "fortnightly", None]),
        "billing_periods_per_year": rng.choice([12, 26, 52]),
        "max_length": rng.choice([0, 12, 36, 60, 60]), "min_length": rng.choice([1, 3, 6]),
        "capacity_limit": float(rng.choice([8000, 30000, 90000])),
        "exposure_limit": float(rng.choice([10000, 40000, 120000])),
        "base_rate": rng.choice([6.0, 15.5, 24.0]), "rate_adjustment": rng.choice([0.0, -0.75, None]),
        "support_fee": rng.choice([25.0, 60.0]),
        "risk_margin": rng.choice([0.0, 0.0, 3.5]), "priority_score": rng.choice([0.0, 640.0, 700.0]),
        "workloads": [workload(rng, i) for i in range(rng.randint(0, 8) if n is None else n)],
    }
    if rng.random() < 0.25:
        req["budget_override"] = round(rng.uniform(100, 6000), 2)
    if rng.random() < 0.7:
        req["requested_length"] = rng.choice([1, 12, 24, 36])
        req["requested_amount"] = float(rng.choice([1500, 9000, 30000]))
    return req


def cases(seed=1, count=500):
    rng = random.Random(seed)
    return [(request(rng), policy(rng)) for _ in range(count)]
```

## `sample_request.json`

Use it for `decider build` warm-up and the serving gate. Its params document is the default `POLICY` with
`reserved_ladder` set to the benchmark ladder: `length` 3, 6, 12, 24, 36, 48 and 60, `min_amount` 500,
`max_amount` 900 × length, and no discount or quotes (`spot_ladder` empty).

```json
{
  "tenant_id": "t34167",
  "reserved_requested": true,
  "spot_requested": false,
  "monthly_budget": 4716.67,
  "billed_spend": 2890.46,
  "other_commitments": 0.0,
  "general_allowance": 200.0,
  "cache_allowance": 80.0,
  "billing": "weekly",
  "billing_periods_per_year": 52,
  "max_length": 60,
  "min_length": 3,
  "capacity_limit": 90000.0,
  "exposure_limit": 40000.0,
  "base_rate": 15.5,
  "rate_adjustment": 0.0,
  "support_fee": 25.0,
  "risk_margin": 3.5,
  "priority_score": 640.0,
  "workloads": [
    {
      "id": "w0",
      "host": "legacy",
      "class_code": "C",
      "family": "web",
      "burst": false,
      "migration_cost": 709.05,
      "monthly_cost": 623.72,
      "units": 29.2,
      "quota": 87.8,
      "overdue_amount": 0.0,
      "overdue_age_days": 30,
      "months_left": 0,
      "disputed": false,
      "movable": false
    },
    {
      "id": "w1",
      "host": "legacy",
      "class_code": "D",
      "family": "storage",
      "burst": false,
      "migration_cost": 2352.51,
      "monthly_cost": 307.43,
      "units": 23.4,
      "quota": 115.0,
      "overdue_amount": 786.6,
      "overdue_age_days": 30,
      "months_left": 11,
      "disputed": true,
      "movable": false
    },
    {
      "id": "w2",
      "host": "legacy",
      "class_code": "C",
      "family": "storage",
      "burst": false,
      "migration_cost": 595.94,
      "monthly_cost": 76.98,
      "units": 19.3,
      "quota": 77.7,
      "overdue_amount": 0.0,
      "overdue_age_days": 0,
      "months_left": 11,
      "disputed": false,
      "movable": true
    },
    {
      "id": "w3",
      "host": "vendor",
      "class_code": "C",
      "family": "gpu",
      "burst": false,
      "migration_cost": 1932.85,
      "monthly_cost": 425.32,
      "units": 62.8,
      "quota": 24.4,
      "overdue_amount": 872.35,
      "overdue_age_days": 30,
      "months_left": 0,
      "disputed": false,
      "movable": true
    },
    {
      "id": "w4",
      "host": "vendor",
      "class_code": "C",
      "family": "gpu",
      "burst": false,
      "migration_cost": 756.43,
      "monthly_cost": 611.36,
      "units": 60.4,
      "quota": 22.3,
      "overdue_amount": null,
      "overdue_age_days": 3,
      "months_left": 11,
      "disputed": false,
      "movable": false
    },
    {
      "id": "w5",
      "host": "vendor",
      "class_code": "B",
      "family": "web",
      "burst": false,
      "migration_cost": 654.42,
      "monthly_cost": 394.07,
      "units": 10.1,
      "quota": 117.8,
      "overdue_amount": 0.0,
      "overdue_age_days": 46,
      "months_left": 40,
      "disputed": false,
      "movable": true
    }
  ],
  "requested_length": 1,
  "requested_amount": 1500.0
}
```

Its answer (`reference.decide(sample, {"reserved_ladder": ladder})`):

```json
{
  "best_subset": 32,
  "migrated_ids": [
    "w5"
  ],
  "legacy_migration_cost": 0.0,
  "vendor_migration_cost": 654.42,
  "total_migration_cost": 654.42,
  "legacy_freed": 0.0,
  "vendor_freed": 394.07,
  "burst_freed": 0.0,
  "general_spend": 0.0,
  "cache_spend": 1036.68,
  "protected_overdue": 0.0,
  "general_overdue": 0.0,
  "other_overdue": 0.0,
  "disputed_overdue": 0.0,
  "legacy_spend": 1008.1300000000001,
  "legacy_units": 71.89999999999999,
  "longest_months_left": 11,
  "gpu_units": 0.0,
  "gpu_quota": 0.0,
  "storage_quota": 192.7,
  "vendor_commitment": 220.7723076923077,
  "committed_spend": 2496.39,
  "committed_total": 3504.52,
  "free_budget": 1212.15,
  "max_amount_score": 32400.0,
  "headroom": 31745.58,
  "monthly_saving": 88.71230769230783,
  "has_offer": true,
  "has_fit": true,
  "fit_exact": true,
  "offer_length": 36,
  "offer_min_amount": 1154.42,
  "offer_max_amount": 32400.0,
  "offer_min_charge": 64.73,
  "offer_max_charge": 1140.19,
  "offer_rate": 15.5,
  "offer_surcharge": 0.0,
  "fit_length": 3,
  "fit_min_amount": 1154.42,
  "fit_max_amount": 2700.0,
  "fit_min_charge": 419.14,
  "fit_max_charge": 946.83,
  "fit_rate": 15.5,
  "fit_surcharge": 0.0
}
```
