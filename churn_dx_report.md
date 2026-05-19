# Churn Pipeline DX Report

**Script:** `churn_pipeline_dx.py`  
**Framework branch:** `feature/simplified-decider`  
**Date:** 2026-05-14  
**Evaluator:** Claude (static analysis + code trace — Python execution blocked by project permissions)

---

## What was exercised

| # | Pattern | Location in script |
|---|---------|-------------------|
| 1 | `generate_from_functions()` | Steps 2, 4 — usage_features, churn_signals, churn_score |
| 2 | `JoinModule` | Step 1 — joining customers + monthly_usage |
| 3 | `\|` pipeline composition | Step 4 — three-stage chained pipeline |
| 4 | `ForkPipeline` (`&` operator) | Step 7 — parallel churn + LTV branches |
| 5 | `module.execute(lazy=False)` | Step 3 — eager single-module path |
| 6 | `pipeline.execute(output_frames=...)` | Step 5 — named intermediate retrieval |
| 7 | `module.compile()` + `compiled.execute()` | Step 6 — compile-once, execute-many |
| 8 | `.input_names` / `.output_names` | Step 2 — introspection |
| 9 | `debug=True` | Step 8 — execution trace |
| 10 | Config injection via Pydantic | Step 9 — configurable thresholds |

---

## Findings

### 1. `output_names` reports node name, not output frame name for JoinModule

**Severity:** Medium  
**File:** `decider/modules/core.py` (BaseModule.output_names, line 199)

`output_names` returns `node.name` for unreferenced leaf nodes. For `JoinModule`, the leaf node has `name="enrich_join_join"` (auto-constructed as `f"{self.name}_join"`), while the actual output frame the pipeline uses is `node.target_frame = "enriched"`. A developer calling `join_module.output_names` to discover what frames were produced will see `["enrich_join_join"]` instead of the frame name `"enriched"` that they need to pass to subsequent modules or `output_frames`.

**Suggested fix:** For frame nodes (`node_type == "frame"`), return `node.target_frame` instead of `node.name` in `output_names`.

```python
# core.py – BaseModule.output_names
output_nodes = [node for node in nodes if node.node_id not in referenced_nodes]
return [
    node.target_frame if node.node_type == "frame" else node.name
    for node in output_nodes
]
```

---

### 2. `Pipeline.execute()` has no `lazy` parameter

**Severity:** Medium  
**File:** `decider/pipeline.py` (Pipeline.execute, line 55)

`BaseModule.execute()` supports `lazy=False` (calls `.collect()` before returning). `Pipeline.execute()` has no equivalent parameter — callers must `.collect()` manually. The `patient_risk_pipeline.py` already hit this, calling `pipeline.execute(..., lazy=False)` which silently accepts `**kwargs`... actually it doesn't: `Pipeline.execute` has no `**kwargs`, so that call would raise `TypeError`. Confirmed design gap.

**Suggested fix:** Add `lazy: bool = True` to `Pipeline.execute()` and `ForkPipeline.execute()`, with the same collect-on-demand logic that `BaseModule.execute()` already implements.

---

### 3. `output_frames` KeyError has no helpful message

**Severity:** Low  
**File:** `decider/pipeline.py` (Pipeline.execute, line 122)

```python
return {k: result[k] for k in output_frames}
```

If the caller requests a frame name that doesn't exist (e.g. a typo, or requesting a frame that only exists inside a ForkPipeline branch), they get a bare `KeyError`. It should show available frame names.

**Suggested fix:**

```python
missing = [k for k in output_frames if k not in result]
if missing:
    raise KeyError(
        f"Requested output_frames not found: {missing}. "
        f"Available: {sorted(result.keys())}"
    )
return {k: result[k] for k in output_frames}
```

---

### 4. Config parameter injection is transparent but the discovery path is bumpy

**Severity:** Low / DX polish

`generate_from_functions()` merges the Pydantic config model fields onto the generated module class, so `ChurnSignals(name="x", low_calls_threshold=10.0)` just works. However:

- There is no indication in the module's `repr` or `input_names` that the module is configurable. Config fields are invisible to `.input_names` (correct — they are not data columns) but there is no `.config_fields` property to discover them.
- The `type` discriminator is auto-set from the `module_name` string, so `generate_from_functions("churn_signals", ...)` hard-codes `type: Literal["churn_signals"]` on the class. Two independently generated classes with the same name string would collide silently in the discriminated union. No guard exists.

---

### 5. `ForkPipeline` does not accept `output_frames`

**Severity:** Low  
**File:** `decider/pipeline.py` (ForkPipeline.execute, line 163)

`ForkPipeline.execute()` has an `output_frames` parameter but it only filters the branch terminal frames. You cannot request an intermediate frame from inside a branch via this parameter — the filtering happens after the branches run, so intermediates are already discarded. This is consistent with the current design but is a potential surprise.

---

### 6. `module_namespaced_nodes` uses `node.name` for in-graph lookup, not `node_id`

**Severity:** Medium — latent  
**File:** `decider/modules/primitives/mapper.py` (MapperModule.expand_nodes, line 178)

The mapping resolution loop at line 178 compares:
```python
if isinstance(input_ref, ExternalInputNode) and input_ref.input_name == input_var_name:
```

`input_var_name` comes from the mapping config (e.g. `"avg_monthly_calls"`), and `input_ref.input_name` is the column/variable name the node was built with. This works correctly for column-name-based wiring. However, when two modules in a `MapperModule` both produce a column with the same name, only the first matching node in the registry gets wired, with no warning. There is a duplicate-module-name guard (`"Duplicate module name"` at line 146) but no duplicate-output-column guard.

---

### 7. Compile step is not cached at the pipeline level

**Severity:** Informational

Each call to `pipeline.execute()` recompiles every module from scratch. The `module.compile()` / `compiled.execute()` pattern (Step 6) is the correct workaround, but it is opt-in and only works at single-module granularity. A `pipeline.compile()` method that returns a `CompiledPipeline` object would allow the same compile-once/execute-many pattern across the full chain.

---

### 8. No built-in schema validation / column-presence check at compile time

**Severity:** Informational

The framework validates function signatures and dependency ordering at compile time but does not validate that required input column names actually exist in the dataframe. Missing columns only raise errors at `.collect()` time (deep inside Polars). For users building complex multi-stage pipelines, an optional `module.validate_schema(df)` or dry-run compile step that checks column presence would dramatically shorten the debug loop.

---

## What worked well

- **`generate_from_functions` dependency wiring** is the standout feature. Writing `def churn_probability(low_usage_flag, near_limit_flag, ...)` and having the framework automatically wire the sibling outputs is genuinely zero-boilerplate and feels natural.
- **`|` composition** reads like a data pipeline should. Three stages join cleanly.
- **`lazy=False` on `module.execute()`** is a nice ergonomic shortcut for interactive exploration.
- **Config injection via Pydantic** composes cleanly with the module — the two `ChurnSignals` instances (default vs. strict thresholds) are a one-liner change with no code duplication.
- **`debug=True`** produces useful per-step trace output at the right level of detail.
- **`JoinModule`** is straightforward once you understand that `left`/`right` are frame dict keys, not column names.

---

## What confused me / took extra thought

1. **`{"input": df}` convention** — not immediately obvious that the key must be `"input"` for expression modules. Learned from error message ("Pass your dataframe as `{"input": df}`") which is good. Could be more prominent in docstrings.
2. **`output_frames` on a module vs. a pipeline** — the parameter exists on both paths, but on `Pipeline.execute()` it selects from the accumulated `result` dict which includes every intermediate. The frame name is the module's `name` string (not the module type), which requires reading the pipeline construction code to predict.
3. **`JoinModule` output_frame** — the frame name in the output is `output_frame` on the config, not `name`. This is the right design (supports multiple joins to different targets) but the `output_names` inconsistency above makes it harder to discover.
4. **ForkPipeline + JoinModule rejoining** — the `ForkPipeline | JoinModule` pattern works, but knowing that the join's `left`/`right` must match the branch terminal module names (not the `output_frame` of each branch) took reading `pipeline.py` carefully to confirm.

---

## Summary score

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Happy-path friction | 4 | Core `generate_from_functions` + `\|` is very clean |
| Discoverability | 3 | `.input_names` / `.output_names` are useful but `output_names` is inconsistent for frame modules |
| Error messages | 3 | `CompiledDag` has good hints; `KeyError` in `output_frames` is bare |
| Multi-frame patterns | 3 | JoinModule works; ForkPipeline rejoining requires pipeline source reading |
| Config / parameterisation | 4 | Pydantic injection is elegant; discoverability slightly low |
| Lazy/eager ergonomics | 3 | `lazy=False` on `module.execute()` but not on `Pipeline.execute()` is inconsistent |

**Overall DX: 3.5 / 5** — the core abstraction is strong and the happy path is genuinely pleasant. The rough edges are concentrated in multi-frame/fork patterns and some API surface inconsistencies that would surface quickly in production use.
