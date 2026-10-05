# Credit and data-science functions

Credit and data-science functions from the team's credit library are going to merge into decider soon. This note covers only **where each kind of function fits** in the redesigned API, so the layout leaves room for them now. How and what gets merged is out of scope. It is decided function by function at merge time, and much of that library is proprietary, so nothing specific to it belongs in this repo before then.

## Four kinds, four homes

Credit functions come in four shapes. Each has an obvious home in the layout:

| Shape | Typical examples | Home | How a user reaches it |
|---|---|---|---|
| **Per-record function:** one record's values in, a value out, pure | score ↔ probability-of-default conversion, WoE or points lookup, credit ratios | **`decider.ops.credit`** (`kernel-functions.md`) | called inside any step; runs in Python and in kernels; tunables are the calling step's `param()`s |
| **Fitting from data:** a frame in, a document out (offline, never on the serving path) | binning, WoE/IV tables, feature selection, target construction, profiling | **`decider.credit`** (a library package) | produces documents decider already runs: scorecard bins (`ScorecardConfig`), `param_table` rows, table and tree documents |
| **Population metric:** distributions in, a number or table out | Gini, AUC, PSI | **`decider.experiments.metrics`** built-ins, plus data checks and `diff` summaries | `experiment.json` names them, reports plot them, `checks` flags drift, and `diff.outputs` reports PSI between versions |
| **Infrastructure:** data access, config, plots, notebook tools | | **stays outside decider** | data sources become `DatasetProvider` client classes named in `datasets/config.json` (`notes/datasets.md`); plots use the `Plot` registry |

### What it looks like

The names below are illustrative.

```python
from decider import ScorecardConfig, flow, param, step
from decider.ops import credit

card = ScorecardConfig.fit(train, target="bad", features=["income", "utilisation"], binning="quantile")

@step
def pd(score: float, offset: float = param(600.0), factor: float = param(20.0)) -> float:
    return credit.score_to_pd(score, offset, factor)

pipeline = flow(card, pd, name="application")
```

```python
from decider.experiments import Experiment
from decider.experiments.metrics import Gini, Psi
Experiment("refit", pipeline, holdout).scenario("v2", ...).metric(Psi(column="pd")).metric(Gini(score="pd", target="bad"))
```

## Rules

1. **No new step constructors.** A credit function becomes part of a flow by being called from a step, or by producing a document an existing step runs. Fitting extends the existing `ScorecardConfig` with a `fit(...)` classmethod rather than adding a new primitive ("extend, not new primitives").
2. **`decider.ops.credit` follows the kernel contract** in `kernel-functions.md`: pure, scalar, `@register_jitable`, the same answer in every mode, and covered by the cache salt. A function that can't meet it isn't an `ops` function. It goes in `decider.credit`, or runs as a `frame_step`.
3. **Fitting is frame-level polars, never on the serving path.** It sits in the libraries layer (L3) beside `experiments` and `checks`. It may import `steps.scorecard` to build documents; the core never imports it.
4. **Heavy dependencies stay optional.** Scientific and optimisation libraries, and any compiled frame-level extension (a polars plugin, for example), go in a `decider[credit]` extra that is imported lazily. `import decider` and the serving path stay as light as they are today. Compiled frame-level code is fine for metrics and fitting, but it never runs inside a numba kernel.
5. **Polars first.** Merged functions take polars frames (for fitting and metrics) or scalars (for `ops`). Other dataframe libraries are converted at the edge, not supported throughout.
6. **What stays proprietary plugs in, rather than merging.** A private package can ship any of these:
   - `Metric`, `Check`, `DatasetProvider` and `ConfigurableStep` subclasses, discovered through registry entry points (WP 5.4);
   - kernel functions under the same contract, safe once `fingerprint` hashes `module.attr` calls (kernel-functions finding 4).

   Decider never needs the private code in its own tree.
7. **Porting follows `CLAUDE.md`.** That means ponytail ultra, behaviour rather than text, and tests ported alongside into `tests/ops/`, `tests/credit/` and `tests/experiments/`. Each function is checked against the original on reference data before the original is retired.

## Where it lands in the plan

Phase 7 starts after phases 5 and 6, which provide `decider.ops`, the metric and check registries, and entry points. It is not part of the API refactor itself. It is listed here so the refactor leaves the right holes:

| WP | Scope |
|---|---|
| 7.1 | `decider.ops.credit`: per-record functions |
| 7.2 | Gini, AUC and PSI as experiment `Metric`s, plus a drift data check and a PSI summary in `diff.outputs` |
| 7.3 | `decider.credit` fitting, and `ScorecardConfig.fit` |
| 7.4 | the `decider[credit]` extra and lazy imports |

The user picks which functions move, per work package.

## Questions for the user

1. **Two names: `decider.ops.credit` for per-record functions and `decider.credit` for fitting.** The split follows the rule that `ops` means "callable inside a step, in kernels". Alternatives for the fitting package: `decider.modelling`, or putting fitting under `decider.steps.scorecard` (which is too narrow for feature selection and targets).
2. **Fitting entry point.** `ScorecardConfig.fit(frame, ...)` is the recommendation, with lower-level `decider.credit` functions for custom work. The alternative is `decider.credit` returning a document that you then load.
