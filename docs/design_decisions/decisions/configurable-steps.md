# Configurable steps: trees, decision tables and flow files

Read this before changing how tree or table documents become kernels, or what a flow file may do; the mechanics are in `docs/architecture/configurable-steps/`.

## A tree or table is data walked by one shared njit function

- `to_ir` emits one `row` node whose `fn` (`walker.walk`, `matcher.match`) is the same for every tree and table; the document becomes int64/float64/byte arrays.
- Why: editing a threshold, row or shape never recompiles, and there is no line cap or per-shape compile (codegen was unpredictable at depth 10 and hit the 100-level indent limit).
- See: [runtime arrays vs generated code](../experiments/runtime-arrays-vs-generated-code.md), `decider/steps/trees/_walker.py`, `decider/steps/tables/_matcher.py`

## Consts reach the kernel as addresses, not arrays

- `Program.consts` holds array addresses read by the shim's `load_i64/f64/u8`; the node's `reference` keeps the arrays alive.
- Why: passing arrays cost 525 ns/row in reference counting on a one-node tree, 30 ns by address.
- See: [tree walker costs](../experiments/tree-walker-costs.md), `docs/architecture/configurable-steps/index.md`

## Inputs are grouped float/int/bool/span; outputs are picked by an intrinsic

- Grouped inputs let an `overload` split the row tuple with static slices (numba cannot index a mixed tuple); `walker.pick` emits one call per output, and `walk` has `no_cpython_wrapper`.
- Why: type-level or `forceinline` recursion over outputs took 48 s at 8 outputs or hit `RecursionError`; the intrinsic compiles the walker for 521 outputs in 25 s.
- See: [tree walker costs](../experiments/tree-walker-costs.md), `docs/architecture/configurable-steps/trees.md`

## A wide `mode: "all"` tree is several row nodes

- A tree with more than `part_outputs` outputs (rules times columns; 128 by default, none for never) is cut into row nodes `<name>/part_<k>` of at most that many, each with whole rules; a rule keeps its `<rule>.<column>` names. A tree with params stays one node, since a param is numbered within its node and named after its path.
- Why: the fused kernel's compile time grows with the square of the outputs of one call (521 rules: 196 s cold, 14 s split in 32s), and narrow kernels are shared. The walker's own `pick` is not the cause.
- Why 128: every part costs a single record 15-20 us, so parts of 32 made `score()` 11-14% slower than an unsplit tree; at 128 it is within 3-6% and a 521-rule tree binds cold in under 40 s. Parts of 32 are 1.4-1.5x faster for a big frame, so `part_outputs` is a setting. See [part size](../experiments/tree-split-part-size.md).
- Each part reads every feature with the kind the whole tree gives it. A node breakpoint `<name>#<id>` matches the parts under `<name>`.
- Cost: `<name>` alone no longer names one node, so a trace, a diff and `checks` report per part (`<name>/part_1`); `_split.MAX_OUTPUTS` is the default width.
- See: [wide output compile](../experiments/wide-output-compile.md), `decider/steps/trees/_split.py`

## Every tree format validates into one `Tree` graph

- v3, `flat_rule` and `prioritized_flat_rule` all become `Tree` (a DAG; unreachable nodes dropped); `TreeConfig` stores the document so the editor keeps node positions.
- Flat nodes get path ids (`"0.1"`), not `uuid4`, so breakpoints and cache keys are stable; thresholds are `Value[int | float]` so int64 ids above 2**53 stay exact.
- No version marker means v3 (a real v1 always carries one); v0-v2 raise; Python function references (`output_fn` and kin) are rejected by name.
- See: `decider/steps/trees/schema/`, `docs/architecture/configurable-steps/trees.md`

## Nulls follow decider_old unless `null_handling="error"`

- A null comparison is unknown and takes the otherwise branch; each comparison row carries an `UNKNOWN` target, so no Kleene state and no duplicated rows.
- A numeric null is a fill (`NaN`, `-2**63`), not an Optional; a NaN in the data is a null.
- Why: `T | None` cost a mask array per feature per launch (`score()` p50 63 -> 102 us; 58-61 us with fills).
- See: [tree walker costs](../experiments/tree-walker-costs.md), `docs/architecture/configurable-steps/trees.md`

## Strings are matched as bytes; regex, case folding and trimming run in Python

- A string feature is a `bytes | None` span input; string outputs are `Literal` indices decoded after the kernel; nothing is dictionary-encoded.
- Why: numba has no regex engine or Unicode folding tables, so those trees fall back per step (`strict_compile=True` raises).
- See: `docs/architecture/configurable-steps/trees.md`, `docs/architecture/engine/boundary.md`

## `trace_output` is a Ball-Larus path number

- Each jump carries a weight and the walker sums the weights it takes; the output is a `Literal` of every path's `>`-joined node ids.
- Why: a kernel cannot write a variable-length string, and with shared nodes the leaf does not name the path. Cost: one add per row, within noise; over 100,000 paths is refused.
- See: `decider/steps/trees/_trace.py`

## The reference walkers are independent implementations

- `reference` walks the model in plain Python and shares only feature kinds with the encoder; it is the interpreted mode and the session locator (`<name>#<id>`, `<name>#<row>`).
- Why: tests run every fixture in all three modes, so an encoder bug cannot hide behind a copy of itself.
- See: `decider/steps/trees/_reference.py`, `decider/steps/tables/_reference.py`

## Table rows are inline or from params, with one matcher

- Both sources are the same `Rows` type, so one compiled matcher serves both and changing rows never recompiles; param rows are validated and packed once when the params document arrives.
- `ParamsCache.key` remembers the last document object, so params documents are immutable once passed (rows-param `score()` p50 88.6 -> 35.8 us).
- A String output of param rows must be an `Enum` column (choices are IR); not: running such tables in Python.
- See: [trees and tables vs decider2](../experiments/trees-tables-vs-decider2.md), `docs/architecture/configurable-steps/tables.md`

## Table semantics are decider_old's, with band ladders per `eq` group

- First match wins, else `default`; a comparison with a null never matches (a null `eq` value is not a wildcard, unlike decider2).
- Rows with equal `eq` values form one ladder for open edges, neighbour fill and contiguity; `in` lists do not split ladders (that would silently open edges).
- Not built: a matched-row output (the table's `path_output`); needs a new output kind in `pick` and the reference.
- See: `docs/architecture/configurable-steps/tables.md`

## Flow files are a front end to `flow` / `branch` / `loop`

- `decider.steps.flowfile.load_flow` builds the same `Step` objects written by hand, so the engine and tooling change nothing; `flow: name` loads `flows/name.*` by reference and conditions are a column, a component or the closed `{expr: ...}` grammar.
- A `step:` name matching both a component and a config document is an error, never a precedence rule; a `branch` with no `else` is one-arm.
- `a OP b` with no constant side compiles as `(a) - (b) OP 0` in float, so exact equality of large ints can be off.
- See: `decider/steps/flowfile/_load.py`, `docs/architecture/configurable-steps/expr-and-flowfile.md`

## Pipeline structure lives with the code, not in a config version

- A flow file is the only copy of the structure, deployed with the components it names; params and step documents stay versioned in `configs/`.
- Not: a `FlowConfig(ConfigurableStep)` inside a config version. It would reverse "structure is Python", which exists because decider_old's structure-in-config drifted into copies.
- See: `docs/architecture/configurable-steps/expr-and-flowfile.md`, `docs/architecture/config-store.md`
