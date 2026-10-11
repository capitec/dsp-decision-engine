# Configurable steps: trees, decision tables and flow files

Read this before changing how tree or table documents become kernels, or what a flow file may do; the mechanics are in `docs/architecture/configurable-steps/`.

## A tree or table is data walked by one shared njit function

- `to_ir` emits one `row` node whose `fn` (`walker.walk`, `matcher.match`) is the same for every tree and table; the document becomes int64/float64/byte arrays.
- Why: editing a threshold, row or shape never recompiles, and there is no line cap or per-shape compile (codegen was unpredictable at depth 10 and hit the 100-level indent limit).
- Not: Rust (PyO3 or C ABI), a polars plugin, ZEN/JDM or a `@cfunc` pointer table. Rust is 12-35% faster per row but is a second toolchain, loses the numba disk cache unless addresses are arguments and aborts on a panic; the plugin costs 236 µs for one record and 252 crates; ZEN `evaluate_batch` is 1.55x slower; a cfunc interpreter is 8.8x generated code on nested loops.
- Not: polars `when/then` chains for scoring (one record costs 419 µs to 7.8 ms, because building the physical plan rather than the optimizer is the cost, so plan caching cannot close it) or a table of njit callables (28-41x per row).
- The walker's helpers are `inline="always"`: a real call per row between the kernel and the walker doubled the tree's cost (247-262 against 125-133 ns/row); see [tree walker costs](../experiments/tree-walker-costs.md).
- Not built, from ZEN: a per-column bitset index for tables of 8+ rows and a `collect` hit policy (every matching row, not the first).
- See: [runtime arrays vs generated code](../experiments/runtime-arrays-vs-generated-code.md), [alternatives to a numba tree walker](../experiments/alternatives-to-a-numba-tree-walker.md), `decikit/steps/trees/_walker.py`, `decikit/steps/tables/_matcher.py`

## Consts reach the kernel as addresses, not arrays

- `Program.consts` holds array addresses read by the shim's `load_i64/f64/u8`; the node's `reference` keeps the arrays alive.
- Why: passing arrays cost 525 ns/row in reference counting on a one-node tree, 30 ns by address.
- See: [tree walker costs](../experiments/tree-walker-costs.md), `docs/architecture/configurable-steps/index.md`

## Inputs are grouped float/int/bool/span; outputs are picked by an intrinsic

- Grouped inputs let an `overload` split the row tuple with static slices (numba cannot index a mixed tuple); `walker.pick` emits one call per output, and `walk` has `no_cpython_wrapper`.
- Why: type-level or `forceinline` recursion over outputs took 48 s at 8 outputs or hit `RecursionError`; the intrinsic compiles the walker for 521 outputs in 25 s.
- See: [tree walker costs](../experiments/tree-walker-costs.md), `docs/architecture/configurable-steps/trees.md`

## A wide `mode: "all"` tree is several row nodes

- A tree with more than `part_outputs` outputs (rules times columns; 64 by default, none for never) is cut into row nodes `<name>/part_<k>` of at most that many, each with whole rules; a rule keeps its `<rule>.<column>` names. A tree with params stays one node, since a param is numbered within its node and named after its path.
- Why: the fused kernel's compile time grows with the square of the outputs of one call (521 rules: 196 s cold, 14 s split in 32s), and narrow kernels are shared. The walker's own `pick` is not the cause.
- Why 64: a single record, start-up and a big frame pull the width in different directions, and 64 is the best compromise over ten rule shapes (table below; [part size](../experiments/tree-split-part-size.md)). A tree of up to 64 rules is never split. From 250 rules up a single record is within 2-3% of an unsplit tree, as with 128; only a tree of 65 to 128 rules pays (up to 8%), and its author can set `part_outputs: 128`. Parts of 128 stage twice as slowly per process and make a big frame 1.4-1.5x slower than parts of 32 (64: 1.1x). A user who runs large frames sets 32; one who needs the last few percent and has a cold start to spare sets more or none.
- Each part reads every feature with the kind the whole tree gives it. A node breakpoint `<name>#<id>` matches the parts under `<name>`.
- Cost: `<name>` alone no longer names one node, so a trace, a diff and `checks` report per part (`<name>/part_1`); `_split.DEFAULT_PART_OUTPUTS` is the default width. Every process compiles its fused parts again: `examples/transaction-fraud` stages in 61 s at parts of 128, 31 s at 64 and 21 s at 32, for no measurable change in its `run(record)`.
- See: [wide output compile](../experiments/wide-output-compile.md), `decikit/steps/trees/_split.py`

Measured by the part-size sweep, median over ten rule shapes (`run(record)` against an unsplit tree; cold bind; `run()` over 200k rows against parts of 32):

| Width | `run(record)` at 250-521 rules | `run(record)` at 100 rules | cold bind, 250-521 rules | `run()` over a frame |
|---|---|---|---|---|
| 32 | 1.11-1.13x | 1.18x | 22-34 s | 1.0x |
| 64 (default) | 1.02-1.03x | 1.08x | 16-17 s | 1.1x |
| 128 | 0.98-1.03x | not split | 26-34 s | 1.4-1.5x |

## Every tree format validates into one `Tree` graph

- v3, `flat_rule` and `prioritized_flat_rule` all become `Tree` (a DAG; unreachable nodes dropped); `TreeConfig` stores the document so the editor keeps node positions.
- Flat nodes get path ids (`"0.1"`), not `uuid4`, so breakpoints and cache keys are stable; thresholds are `Value[int | float]` so int64 ids above 2**53 stay exact.
- No version marker means v3 (a real v1 always carries one); v0-v2 raise; Python function references (`output_fn` and kin) are rejected by name.
- See: `decikit/steps/trees/schema/`, `docs/architecture/configurable-steps/trees.md`

## Nulls follow decider_old unless `null_handling="error"`

- A null comparison is unknown and takes the otherwise branch; each comparison row carries an `UNKNOWN` target, so no Kleene state and no duplicated rows.
- A numeric null is a fill (`NaN`, `-2**63`), not an Optional; a NaN in the data is a null.
- Why: `T | None` cost a mask array per feature per launch (`run(record)` p50 63 -> 102 us; 58-61 us with fills).
- See: [tree walker costs](../experiments/tree-walker-costs.md), `docs/architecture/configurable-steps/trees.md`

## Strings are matched as bytes; regex, case folding and trimming run in Python

- A string feature is a `bytes | None` span input; string outputs are `Literal` indices decoded after the kernel; nothing is dictionary-encoded.
- Why: numba has no regex engine or Unicode folding tables, so those trees fall back per step (`strict_compile=True` raises).
- Not: a C (PCRE2) or Rust regex called from the kernel. It saves 20-40 ns/row at most (lazy at the node against a column pass), PCRE2 backtracks (`^(a+)+$` is 211 µs/row), and an unprotected Rust panic aborts the process.
- See: [string matching beyond equality](../experiments/string-matching-beyond-equality.md), See: `docs/architecture/configurable-steps/trees.md`, `docs/architecture/engine/boundary.md`

## `trace_output` is a Ball-Larus path number

- Each jump carries a weight and the walker sums the weights it takes; the output is a `Literal` of every path's `>`-joined node ids.
- Why: a kernel cannot write a variable-length string, and with shared nodes the leaf does not name the path. Cost: one add per row, within noise; over 100,000 paths is refused.
- See: `decikit/steps/trees/_trace.py`

## The reference walkers are independent implementations

- `reference` walks the model in plain Python and shares only feature kinds with the encoder; it is the interpreted mode and the session locator (`<name>#<id>`, `<name>#<row>`).
- Why: tests run every fixture in all three modes, so an encoder bug cannot hide behind a copy of itself.
- See: `decikit/steps/trees/_reference.py`, `decikit/steps/tables/_reference.py`

## Table rows are inline or from params, with one matcher

- Both sources are the same `Rows` type, so one compiled matcher serves both and changing rows never recompiles; param rows are validated and packed once when the params document arrives.
- `ParamsCache.key` remembers the last document object, so params documents are immutable once passed (rows-param `run(record)` p50 88.6 -> 35.8 us).
- A String output of param rows must be an `Enum` column (choices are IR); not: running such tables in Python.
- See: [trees and tables vs decider2](../experiments/trees-tables-vs-decider2.md), `docs/architecture/configurable-steps/tables.md`

## Table semantics are decider_old's, with band ladders per `eq` group

- First match wins, else `default`; a comparison with a null never matches (a null `eq` value is not a wildcard, unlike decider2).
- Rows with equal `eq` values form one ladder for open edges, neighbour fill and contiguity; `in` lists do not split ladders (that would silently open edges).
- Not built: a matched-row output (the table's `path_output`); needs a new output kind in `pick` and the reference.
- See: `docs/architecture/configurable-steps/tables.md`

## A table variable is a float unless the document declares it `int`

- `variable_types: {"channel": "int"}` makes the variable an int64 input compared exactly (`eq`, `in`, `between`, integer columns only; null is `-2**63`); a step that writes it must write an `int`. Without it every numeric variable is a float, exact up to 2**53, and a document without `variable_types` compiles as before.
- Why: a code above 2**53 (an identifier, `2**60`) cannot be told from its neighbour as a float.
- Not: inferring it from the columns (an input read only by `eq`/`in` over integer columns): it silently changed which type a step had to write for a document already in use, and a float input was then truncated. Built and reverted.
- See: `tests/tables/test_table_int_variables.py`, `docs/architecture/configurable-steps/tables.md`

## Flow files are a front end to `flow` / `branch` / `loop`

- `decikit.steps.flowfile.load_flow` builds the same `Step` objects written by hand, so the engine and tooling change nothing; `flow: name` loads `flows/name.*` by reference and conditions are a column, a component or the closed `{expr: ...}` grammar.
- A `step:` name matching both a component and a config document is an error, never a precedence rule; a `branch` with no `else` is one-arm.
- `a OP b` with no constant side compiles as `(a) - (b) OP 0` in float, so exact equality of large ints can be off.
- See: `decikit/steps/flowfile/_load.py`, `docs/architecture/configurable-steps/expr-and-flowfile.md`

## Pipeline structure lives with the code, not in a config version

- A flow file is the only copy of the structure, deployed with the components it names; params and step documents stay versioned in `configs/`.
- Not: a `FlowConfig(ConfigurableStep)` inside a config version. It would reverse "structure is Python", which exists because decider_old's structure-in-config drifted into copies.
- See: `docs/architecture/configurable-steps/expr-and-flowfile.md`, `docs/architecture/config-store.md`

## A custom step's data is its fields; its logic is a staticmethod

- `CustomFunction` writes `to_ir` for a `ConfigurableStep` whose fields feed one pure function, `impl`. Each `impl` argument has a field of the same name: a `str` field is the column to read, a `Value[T]` field is a value.
- Why fields: a step that reads a module-level object or a closure hides its data from numba, from the structure diff and from the config. Fields dump, load, diff and compile like any other config.
- Why a `staticmethod`: with no `self` numba compiles `impl` as a plain function, and the diff fingerprints its body.
- Why checks at class creation: a missing `impl`, an argument without a field, or a field that is neither a column nor a `Value` would otherwise surface as a wiring error far from the class. Rows, objects and lists are `param_table` or a hand-written `to_ir`, not a field type this base accepts.
- No `type` tag is derived from the class name: without a `Literal` tag the class already loads by import path, and a derived alias could collide across packages.
- See: `decikit/steps/_custom.py`, `tests/steps/test_custom_function.py`
