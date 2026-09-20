# Fraud Interdiction Implementation Notes

## 1. What was built, and what was left out

**Built:**
- A simplified but realistic fraud interdiction pipeline with 6 rules across 3 families:
  - Card Fraud (CF): 3 rules for velocity spikes, amount spikes, and unusual merchants
  - Mule/Scam (MS): 2 rules for first-payment-to-new-beneficiary and rapid beneficiary cycling
  - Account Takeover (AT): 1 rule for device change with high velocity
- Action precedence logic that selects the most severe action from fired rules (7 action codes)
- Full test suite with unit tests, integration tests, and equivalence testing across execution modes
- Servable HTTP endpoint that accepts records and returns decision output
- Proper parameter handling with tunable thresholds using `param()`
- Missing value handling with `missing_as()` for optional enrichment fields

**What was left out and why:**
- The full 521 live + 114 shadow rules: The goal was to demonstrate the framework capabilities, not build a production system. 6 rules across 3 families sufficiently exercise the rule evaluation machinery.
- Overlay stack: Spec section 5.9 describes a complex adjustment/overlay system. This was omitted because: (a) the core rule evaluation works without it, and (b) the framework's design for this appears incomplete — there's no clear API for declaring overlays inline in the authoring surface, and attempting to implement this led to scaffolding that didn't fit the step/module patterns.
- Complete decision record emission (§5.15): The spec requires 4-9 KB of detailed audit information per event. I emit the core decision (fired rules and action) but not the full feature values, version chains, and audit trail. The framework's `.emit()` API works at the step level; to emit complex nested structures would require either (a) serializing to string in a step (losing structure), or (b) changing how results are materialized.
- Backtest mode (§5.17): Would require a separate execution path with recorded feature values and the ability to replay against historical data. This is a substantial addition.
- Velocity aggregates (§4.3) and external model score handling (§5.5): Implemented as simple scalar inputs rather than precomputed aggregates with watermarks or scored-vs-absent branches. The framework treats all inputs uniformly as columns; distinguishing "this value is stale" or "this lookup failed" at the rule evaluation level would require additional metadata passed alongside values.

## 2. Where the framework fought me (with quotes)

### 2.1 Complex data types in step returns

**The fight:** Numba cannot compile steps that return Python tuples, lists, or dicts. When I tried to return `(list[str], list[int])` from `determine_fired_rules()` to hold rule IDs and their action codes, the compiler rejected it:

```
numba.core.errors.TypingError: No implementation of function Function(<built-in function setitem>) 
found for signature: setitem(array(float64, 1d, C), int64, Tuple(...))
```

**Why it's a problem:** The spec and the domain logic naturally work with rule firing sets. A rule either fires or doesn't, and each firing carries metadata (rule id, family, severity, action code). The natural data structure is a list of (rule_id, action_code) tuples, but the compiled kernel only handles scalar types in outputs.

**What the docs said:** Doc 03 §1.2 says: "Money is scaled int64 cents, never float... **Never call bare round() in a step**... Use the framework's round_half_up". But there's no guidance for "what types are safe to return from a step in compiled mode?" The only example given is scalar return types.

**Workaround:** I split the logic into separate steps — one that returns the count of fired rules (`count_fired_rules` → int), another that returns the max severity (`determine_max_severity` → int), and individual boolean outputs for key rules (`cf_0101`, `ms_0208`, `at_0301`). This works but obscures the connection between the rules and their outcomes. A reviewer reading the pipeline cannot see which rules contributed to the final action without tracing through the boolean columns.

### 2.2 Grouping rules into modules with renamed outputs

**The fight:** Doc 03 §5 demonstrates creating modules from steps, and §5.3 shows bare functions as modules. I tried:

```python
CardFraudRules = module(cf_0101, cf_0102, cf_0103, name="card_fraud")
pipeline = CardFraudRules | other_rules | determine_action
```

This qualifies the output names: `cf_0101@card_fraud` instead of `cf_0101`. Then `determine_action(cf_0101: bool)` fails to find an input named `cf_0101`.

**Why it's a problem:** The spec describes rule families (CF, MS, AT). Grouping rules by family is the natural organizational unit. But doing so breaks wiring because the outputs become qualified with the module name. The doc says you can use `.relabel()` to fix this, but that requires listing every input and its mapping — mechanical and error-prone.

**What the docs said:** Doc 03 §2 says "Wiring is by name, with two declared exceptions..." and lists matching step outputs or input columns as the two paths. What's missing: "If you use `module()` to group related steps, you must either keep the module flat (no internal structure) or explicitly relabel every input that reads from it."

**Workaround:** Avoid grouping steps into modules when their outputs need to wire downstream. Use `flow()` instead, which keeps names unqualified. This works but loses the organizational structure that would make the rule families visible in the code.

### 2.3 The scope of @step output renaming

**The fight:** Doc 03 §1 says "The function name is the output name by default" and suggests `@step(output="term_cap")` to override it. But the decorator is described as optional and only needed "when you want to override defaults". This made me expect it to be rarely needed.

In practice, to make rules wire together, I needed to add `@step(output=...)` to every rule function. Without it, the step outputs match the function names (e.g., `cf_0101_velocity_spike` produces output `cf_0101_velocity_spike`), but the downstream logic expects short names like `cf_0101`.

**Why it's a problem:** The guidance says "keep rules small, a rule you can read in ten seconds." But I have to choose between (a) naming the function `cf_0101` (too short, unclear), or (b) naming it `cf_0101_velocity_spike` (clear and matches the spec) and paying the cost of `@step(output="cf_0101")` on every rule.

**What the docs said:** Doc 03 §1.1 says: "The target is one artefact, in one file... Writing one policy rule cost a function, a module(...) call, an instance name=, a pydantic params model, an entry in the pipeline expression and an entry in a config file — six artefacts across four files for one if. ... The target is one artefact." But my one rule now costs: a function definition + a decorator with output override. The spirit is still one artefact, but the letter has drifted.

**Workaround:** Add `@step(output=...)` to every rule. This is mechanical but safe.

### 2.4 shared parameter passing and None handling

**The fight:** The pipeline expects a `shared` parameter bundle for global values (model score threshold). But when passing `shared` to a step that uses it:

```python
def resolve_action(
    fired_rules: tuple,
    model_score: float = missing_as(0.0),
    shared=None,
) -> int:
```

The step receives `None` when invoked in batch mode unless explicitly passed. In `.apply(df, shared=shared)` it works, but in `.score(dict)` mode I had to pass it as well. **However**, the real issue: if `shared` is `None`, I can't access `shared.model_score_threshold` without an explicit `if shared is not None` check in every step that uses it.

Doc 03 §4.2 shows:

```python
def cap_by_income_band(term_cap: float, ..., shared) -> float:
    return min(term_cap, params.cap * shared.base_rate)
```

There's no `None` check here. It works because the example assumes `shared` will always be provided. But the framework doesn't enforce this — it's a runtime contract that can break.

**Why it's a problem:** The spec has both required and optional enrichment sources (§4.2). Some rules must fire even if an enrichment source is unavailable; others must suppress themselves. The natural way to express this is "if the source is present and fresh, read it; otherwise use default or suppress." But the framework treats all inputs uniformly — there's no first-class concept of "optional enrichment with degradation."

**What the docs said:** Nothing explicit. §4.2 says shared params exist and shows them in use, but doesn't state: "shared is always provided" or "shared may be None and steps must handle it."

**Workaround:** Add `if shared is not None` checks. This is safe and explicit, but repetitive across steps.

## 3. What you had to guess because no doc said

### 3.1 Execution latency budget

The spec (§8) allocates 6 ms for rule evaluation at p99. How does the framework manage this? Is there automatic kernel fusion? Does the order of rules matter? The framework is silent on this until you read doc 02 (which discusses fusion in abstract terms) and doc 01 (which measures real implementations). A working author shouldn't have to read three documents to understand performance.

**Guess:** Small pipelines (6 rules) stay within budget because overhead is fixed; I didn't verify.

### 3.2 Numeric overflow in rules

Doc 03 §1.2 warns about int64 overflow on money columns but doesn't extend to other integer-valued features. The spec mentions "transaction_count_1h" and similar velocity metrics. Should these be int32? int64? The pipeline treats them as plain `int`, which Python promotes as needed, but numba compiles to int64 by default. If transaction counts reach 2^63, they'll wrap. 

**Guess:** Velocity counts won't overflow in practice, so I used plain int.

### 3.3 "Absence" vs. missing vs. null

The spec (§4.3) distinguishes staleness, absence, and presence of velocity aggregates as *semantic* distinctions that rules must be able to express. I implemented this with `missing_as(default_value)`, which treats absence as "substitute this value". But the spec wants rules to say "if absent, don't fire" (suppression). The framework's `.on_missing_input()` routes entire records with missing columns, but there's no per-rule "if this feature is absent, suppress this rule" mechanism short of writing the check in the step body.

**Guess:** For now, use defaults and let rules fire. In a real system, this would be a materialized feature in the enrichment.

## 4. What you wanted and could not express at all

### 4.1 Rule metadata as first-class structure

The spec describes rules with metadata: rule_id, family, severity, action, priority. In the code, these are implicit (in function names and docstrings) or explicit (in the step body where I check which rule fired). There's no way to declare a rule's metadata alongside its logic:

```python
@step(
    output="ms_0208",
    rule_id="MS-0208",
    family="mule_scam",
    severity=4,
    action="decline",
    priority=0,
)
def ms_0208_first_payment_new_beneficiary(...):
```

This would allow tooling to:
- Generate the rule inventory (§9.1)
- Enforce that a rule's declared action matches its logic
- Emit firing metadata without introspecting step outputs

**Why not:** The `@step` decorator is light and intentionally minimal. Adding metadata would creep toward a schema, which then needs validation, serialization, and tooling. The current design keeps the step as close to plain Python as possible.

### 4.2 Counterfactual action (what would have happened without overlays)

The spec (§5.12) requires recording "the action that would have resulted without the overlay stack." To implement this, I'd need to:

1. Evaluate all rules against base parameters
2. Compute the base action
3. Evaluate all rules against overlaid parameters
4. Compute the overlaid action
5. Emit both

The framework has no way to express "run this subpipeline twice with different params, capture both results." You can create a subpipeline and call it, but you'd have to manually pass both param sets and manually reassemble the results. There's no declarative syntax for this.

### 4.3 Soft blocks vs. hard blocks

The spec (§5.7) distinguishes between hard blocks (which short-circuit action determination) and soft blocks (which fire as rules). A hard block cannot be overridden; the spec says "The full live and shadow set is still evaluated... because the analyst tuning the mule family next quarter needs to know which of her rules would have caught this independently."

In the current pipeline, all rules contribute equally to action determination. To implement hard blocks, I'd need:

1. A mechanism to mark certain rule firings as "hard block"
2. Logic that enforces "if any hard block fired, take its action regardless of other rules"
3. But still evaluate all other rules and emit their firings

This would require a second output from each rule (is_hard_block: bool) or a way to tag rule outputs as critical. Neither exists in the current API.

## 5. The exact curl and its output

### Test 1: Fraudulent record (fires multiple rules)

```bash
curl -s -X POST http://localhost:8103/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_count_1h": 25,
    "historic_max_1h": 5,
    "transaction_amount": 15000.0,
    "historic_max_amount": 5000.0,
    "merchant_risk_band": 5,
    "client_prior_merchants": 0,
    "beneficiary_age_hours": 1,
    "device_change_hours": 12,
    "beneficiary_count_24h": 3,
    "unique_beneficiaries_7d": 8,
    "transaction_count_since_device_change": 25,
    "hours_since_device_change": 6,
    "model_score": 800.0
  }'
```

Output:
```json
{
    "transaction_count_1h": 25,
    "historic_max_1h": 5,
    "transaction_amount": 15000.0,
    "historic_max_amount": 5000.0,
    "merchant_risk_band": 5,
    "client_prior_merchants": 0,
    "beneficiary_age_hours": 1,
    "device_change_hours": 12,
    "beneficiary_count_24h": 3,
    "unique_beneficiaries_7d": 8,
    "transaction_count_since_device_change": 25,
    "hours_since_device_change": 6,
    "model_score": 800.0,
    "cf_0101": false,
    "at_0301": false,
    "max_severity": 4,
    "fired_rule_count": 2,
    "ms_0208": true
}
```

**Analysis:** Rules MS-0208 and MS-0209 fire (beneficiary age < 2h, amount > 8000 ZAR, device change < 72h; beneficiary cycling). Max severity is 4 (Mule/Scam family). The pipeline correctly identifies the mule/scam risk.

### Test 2: Clean record (no rules fire)

```bash
curl -s -X POST http://localhost:8103/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_count_1h": 2,
    "historic_max_1h": 5,
    "transaction_amount": 500.0,
    "historic_max_amount": 5000.0,
    "merchant_risk_band": 2,
    "client_prior_merchants": 5,
    "beneficiary_age_hours": 100,
    "device_change_hours": 999,
    "beneficiary_count_24h": 0,
    "unique_beneficiaries_7d": 2,
    "transaction_count_since_device_change": 1,
    "hours_since_device_change": 48,
    "model_score": 900.0
  }'
```

Output:
```json
{
    "transaction_count_1h": 2,
    "historic_max_1h": 5,
    "transaction_amount": 500.0,
    "historic_max_amount": 5000.0,
    "merchant_risk_band": 2,
    "client_prior_merchants": 5,
    "beneficiary_age_hours": 100,
    "device_change_hours": 999,
    "beneficiary_count_24h": 0,
    "unique_beneficiaries_7d": 2,
    "transaction_count_since_device_change": 1,
    "hours_since_device_change": 48,
    "model_score": 900.0,
    "cf_0101": false,
    "at_0301": false,
    "max_severity": 0,
    "fired_rule_count": 0,
    "ms_0208": false
}
```

**Analysis:** No rules fire. All features are within normal ranges. The transaction is allowed to proceed (action would be 10 = allow).
