# IR specification (task T1.1), draft 3

Status: **final**. Edit inline; mark feedback with `>>> ... <<<`.
§11 records how the draft-2 and draft-3 comments were handled.

This is the contract every later task builds on. It defines the two layers:
**steps**, which users write and combine, and the **IR**, which the engine
executes.

**Nothing in the framework is called "module".** In decider_old, a "module" was
roughly what we now call a step, so reusing the word would be confusing.

---

## 1. The two layers

```
Steps (what users write)  --to_ir()-->  IR (IRNode tree)  --Engine.bind(mode)-->  Executable
```

1. **Steps** are the authoring layer: plain functions, `@step`, `dag(...)`,
   `flow(...)` / `|`, `branch`, `loop`, and `ConfigurableStep` subclasses such
   as trees. Every step is a subclass of `Step` and can be combined with any
   other.
2. **Every step produces its own expanded IR** through `to_ir(ctx)`. There is
   no intermediate stage in which a config step sits unexpanded in the IR. A
   tree config's `to_ir` returns the tree's IR nodes directly.
3. **The IR is a closed set of four node types:** `CallNode`, `SequenceNode`,
   `BranchNode` and `LoopNode`.
   - The engine implements each execution mode, the debugger and the compiler
     **once per node type**.
   - A new step type, written by anyone, only implements `to_ir()` and never
     needs engine changes.
4. **IR nodes are fully resolved:**
   - relabels are applied;
   - `dag` ordering is sorted into written order;
   - param namespaces are fixed;
   - each node carries its **origin**, meaning which step produced it and where
     that step sits.

   The engine needs no further lookups.
5. **Steps don't know their parent; IR nodes do.** `to_ir` is called once per
   placement, with a context carrying the path. So one step object can be used
   in two pipelines, and each placement gets its own IR with its own paths.
6. **Functions convert automatically** wherever a step is expected. Only
   `func | func` needs an explicit `step()`.
7. **Nothing in either layer imports numba or polars.** `run()` and `session()`
   on `Step` import the engine lazily.

---

## 2. Terminology

| Layer | Class | Made by | Meaning |
|---|---|---|---|
| step | `Step` | — | Abstract base: `name`, `\|`, `named`, `relabel`, `to_ir`, `parameters`, `walk`, `run`, `session` |
| step | `FunctionStep` | `step(fn)`, `@step`, or a plain function | One scalar function |
| step | `FrameStep` | `frame_step(fn)` | `DataFrame -> DataFrame` |
| step | `DagStep` | `dag(a, b, c)` | Members ordered by dependencies |
| step | `SequentialStep` | `flow(a, b, c)`, `a \| b \| c` | Members in written order; later writes win (the waterfall) |
| step | `BranchStep` | `branch(...)` | Condition plus arms |
| step | `LoopStep` | `loop(...)` | Condition plus body plus carries |
| step | `ConfigurableStep` | subclass it (pydantic) | Serialisable, config-driven step (trees, tables, scorecards) |
| IR | `IRNode` | `Step.to_ir()` | Abstract base: `origin`, `children()` |
| IR | `CallNode` | | Calls one function. `kind` is `scalar`, `row` or `frame` |
| IR | `SequenceNode` | | Children in execution order |
| IR | `BranchNode` / `LoopNode` | | Control flow |

Class names are settled as `SequentialStep` (built by `flow(...)` and `|`) and
`DagStep` (built by `dag(...)`).

---

## 3. Steps: the authoring layer

### 3.1 `Step` (base)

```python
class Step(ABC):
    __slots__ = ()
    name: str | None

    @abstractmethod
    def to_ir(self, ctx: "IRContext") -> "IRNode": ...

    def __or__(self, other) -> "SequentialStep": ...
    def __ror__(self, other) -> "SequentialStep": ...
    def named(self, name: str) -> Self: ...
    def relabel(self, *, reads=None, writes=None) -> Self: ...

    def parameters(self) -> "ParamsSchema": ...          # every param this step needs (§5.4)
    def walk(self) -> Iterator[tuple[str, "Step"]]: ...   # authoring tree: (path, step)
    def run(self, data, **kwargs): ...
    def session(self, data, **kwargs): ...
```

- Built-in steps are frozen, slotted dataclasses with `eq=False` (identity
  equality). `ConfigurableStep` is a frozen pydantic model.
- `relabel` is shared logic on the base class. It is stored as `reads` and
  `writes` pairs, and applied by `to_ir`.

### 3.2 `FunctionStep`

```python
@dataclass(frozen=True, slots=True, eq=False)
class FunctionStep(Step):
    name: str
    fn: Callable
    outputs: tuple[str, ...]              # default: (fn.__name__,)
    nogil: bool = False
    reads / writes / bound

    def __call__(self, *args, **kwargs): return self.fn(*args, **kwargs)
    def bind(self, **values) -> "FunctionStep": ...

def step(fn=None, /, *, name=None, output=None, outputs=None, nogil=False) -> FunctionStep
```

- Inputs, params and null policies are harvested from the signature (§5).
- **Multiple outputs:** declare `outputs=("a", "b")` and annotate the return
  type as `tuple[float, bool]`. The annotation must be a tuple of the declared
  length; a mismatch is an error when the step is built.
- **The waterfall:** several rules each declare `output="term_cap"`.

### 3.3 `FrameStep`

```python
def frame_step(fn=None, /, *, name=None, reads=None, writes=None) -> FrameStep
```

- `reads=None` / `writes=None` means **unknown lineage**. The IR node is then a
  barrier: names read after it are checked against its actual output at run
  time instead of statically.
- It always runs in Python/polars. A session pauses before and after it.

### 3.4 `DagStep` and `SequentialStep`

```python
def dag(*steps, name=None) -> DagStep     # one element: returns it unchanged
def flow(*steps, name=None) -> SequentialStep
```

| | `flow` / `\|` | `dag` |
|---|---|---|
| order | as written | by dependencies, sorted in `to_ir` |
| two members write the same name | allowed: later wins; every write is kept as a version | error, suggesting `flow` |
| IR produced | `SequenceNode` | `SequenceNode` (already sorted) |

- `a | b | c` is one `SequentialStep(a, b, c)`. `|` appends to an **anonymous** flow;
  a named flow stays a unit.
- `emit(...)` and `drop(...)` are available on `SequentialStep` and `DagStep`.

### 3.5 `BranchStep` and `LoopStep`

```python
def branch(condition, *arms, modifies, name) -> BranchStep     # bool: 2 arms; int: N arms
def loop(condition, body, *, carries, max_iterations, name) -> LoopStep
```

Arms and bodies can be any step.

### 3.6 `ConfigurableStep`

```python
class ConfigurableStep(Step, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: str                          # defaults to the class import path; see "Type tags" below
    name: str

    @abstractmethod
    def to_ir(self, ctx: IRContext) -> IRNode: ...
```

- **A step like any other**, with one difference: it is pydantic, so it
  serialises and validates.
- **It emits IR directly.** Most implementations are a couple of lines: build
  helper steps and call their `to_ir`, or build a `row` `CallNode` for a tree.
- **"Recreating" a step means making a new config and calling `to_ir` again.**
  There is no separate rebuild mechanism. Configs are frozen, so a changed
  config is always a new object, and only that step's IR is regenerated.
- **Adding or removing a rule is a structural change**: reload the step from
  config and regenerate its IR. Everything else (thresholds, patterns, table
  rows) is a param (§5), so changing it regenerates nothing.
- The existing `decider/modules/core.py::BaseModule` becomes this class.

**Type tags.** Every subclass is addressable by its **import path**, and may also
declare a short alias for convenience:

```python
class TreeConfig(ConfigurableStep):          # {"type": "decider.steps.trees:TreeConfig"}
    ...

class TreeConfig(ConfigurableStep):          # also {"type": "tree"}
    type: Literal["tree"] = "tree"
```

- With no `type` declared, the field defaults to `"<module>:<QualName>"` and is
  checked on load.
- With a `Literal` alias declared, the class is registered under the alias, and
  its import path works too.
- `model_dump()` writes the alias when there is one, and the import path
  otherwise.
- Loading an import path imports the module and **checks that the class is a
  `ConfigurableStep` subclass** before using it. Anything else is rejected, so a
  config can't be used to call arbitrary code.
- Two classes claiming the same alias is an error naming both. (Replacing an
  entry for the same qualified name is allowed, for notebook reloads.)
- Unknown tags get a did-you-mean suggestion drawn from the aliases.

---

## 4. The IR

### 4.1 `IRNode` and `Origin`

```python
@dataclass(frozen=True, slots=True)
class Origin:
    path: str                  # "term/by_sector/cap_private"; unique within one IR
    source: str                # import path of the producing step: "credit.rules:cap_by_income",
                               #   "decider.steps.trees:TreeConfig"
    locator: str | None = None # a position inside a row node, e.g. "n17"

class IRNode(ABC):
    __slots__ = ()
    origin: Origin

    @abstractmethod
    def children(self) -> tuple["IRNode", ...]: ...
```

- `origin` is **serialisable** (plain strings), so events and traces carry it
  as is.
- In a running process, the engine also keeps a `path → Step` map, so tools can
  jump from an IR node to the actual step object. That map is not serialised.

### 4.2 The four node types

```python
@dataclass(frozen=True, slots=True, eq=False)
class CallNode(IRNode):
    origin: Origin
    kind: Literal["scalar", "row", "frame"]
    fn: Callable
    inputs: tuple[Input, ...] | None      # resolved names; None = unknown (frame)
    outputs: tuple[Output, ...] | None
    params: tuple[ParamDecl, ...]
    reference: Callable | None = None     # Python twin for row nodes, for stepping inside
    nogil: bool = False

@dataclass(frozen=True, slots=True, eq=False)
class SequenceNode(IRNode):
    origin: Origin
    children_: tuple[IRNode, ...]
    emits: tuple[str, ...] = ()
    drops: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True, eq=False)
class BranchNode(IRNode):
    origin: Origin
    condition: CallNode
    arms: tuple[IRNode, ...]
    modifies: tuple[str, ...]

@dataclass(frozen=True, slots=True, eq=False)
class LoopNode(IRNode):
    origin: Origin
    condition: CallNode
    body: IRNode
    carries: tuple[str, ...]
    max_iterations: int
```

**How `fn` is called in each `CallNode` kind:**

| kind | Call |
|---|---|
| `scalar` | `fn(**inputs, **params)` |
| `row` | `fn(row, params)`, where `row` and `params` are tuples in declared order and the result is a tuple of outputs |
| `frame` | `fn(df) -> df` |

- One standard driver serves all three kinds: gather the declared inputs, call,
  scatter the outputs.
- Params always arrive as **arguments**: keyword arguments for `scalar` nodes, a
  bundle for `row` nodes. Retuning them therefore never regenerates IR and
  never recompiles.
- For `row` nodes, `reference(row, params, visit)` is the plain-Python twin.
  - In interpreted mode the engine calls it, and `visit("n17")` reports each
    internal node, which is how a session steps into a tree.
  - Compiled modes call `fn`.

### 4.3 `IRContext`

```python
@dataclass(frozen=True, slots=True)
class IRContext:
    path: str                          # path of the parent
    shared: "SharedParams"             # the global shared-param type table (§5.2)

    def child(self, name: str | None) -> "IRContext": ...   # None = transparent
    def origin(self, step: Step, name: str | None = None, locator: str | None = None) -> Origin: ...
```

`engine.to_ir(step)` is the entry point. It creates the root context, calls
`step.to_ir`, and checks the result:
- paths are unique;
- shared param types agree;
- every node has an origin.

---

## 5. Inputs and params

### 5.1 Missing values: per input, error by default

| Declared as | Null in the data |
|---|---|
| `x: float` | **error**, naming the input, the step path and the row count |
| `x: float = missing_as(0.0)` | filled with `0.0` |
| `x: float \| None` | passed through as `None` |

### 5.2 Local and shared params

```python
def cap_by_income(term_cap: float, min_net_salary: float,
                  cap: float = param(48.0, ge=6, le=60),                  # local
                  base_rate: float = param(5.0, shared_key="base_rate"),  # shared
                  ) -> float: ...
```

- **Local** params live under the step's path. **Shared** params live under the
  top-level `"shared"` key.
- **Types must agree globally.** `to_ir` records every shared key's type in one
  table. Declaring the same key with a different type fails:

  ```
  shared param 'base_rate' is declared float by term/cap_by_income and int by pricing/fee
  ```

- **Validation, defaults and missing behaviour belong to each step.** Each step
  gets its own generated pydantic model covering its local params *and* the
  shared keys it uses, and validates its view before the function is called. So
  two steps may put different bounds on the same shared key, and each one
  enforces its own.
  - If a shared key is missing from the params document, each step falls back
    to its **own** default. Nothing is checked globally apart from the type;
    §5.5 covers when each step's validation runs.
- `shared` is a reserved top-level key. No step may be named `shared`.

### 5.3 Config values: a literal or a param reference, chosen per value

```python
class TreeConfig(ConfigurableStep):
    type: Literal["tree"] = "tree"
    tree: TreeDocument                 # node thresholds typed Value[float]

class DecisionTableConfig(ConfigurableStep):
    type: Literal["decision_table"] = "decision_table"
    columns: dict[str, DType]
    rows: TableValue                   # inline rows, or {"table": "prices"}
```

- `Value[T] = T | ParamRef`.
  - `0.7` is a literal.
  - `{"param": "hi_thresh", "default": 0.7}` is a local param.
  - `{"param": "base_rate", "shared": true}` is a shared param.
- `TableValue = DataFrame | TableRef`. A table-valued param **carries its own
  schema** (columns and dtypes), so its rows are validated when they arrive.
  - Rows reach the kernel as runtime arrays, so editing rows or changing how
    many there are costs nothing.
  - Only a schema change regenerates IR.
- Inside `to_ir`, `ctx.value(v)` and `ctx.table(t)` turn each value into either
  a literal or a `ParamDecl`.

### 5.4 What a step needs: `parameters()`

Every step can report every param it needs without running anything. This is
what a config UI, a validator or a "complete the params document" tool reads.

```python
pipeline.parameters()
# {
#   "shared": {"min_ratio": {"type": "float", "default": 0.3, "used_by": ["affordability/affordable"]}},
#   "term/cap_by_income": {"cap": {"type": "float", "default": 48.0, "ge": 6, "le": 60}},
#   "price_by_product": {"prices": {"type": "table",
#                                   "schema": {"product": "str", "rate": "float"}}},
# }
pipeline.parameters().defaults()        # a complete params document of defaults
pipeline.parameters().json_schema()     # JSON Schema, for rendering forms
```

It is implemented once: `to_ir`, then collect `ParamDecl`s from every
`CallNode`. So a step type never implements it separately.

### 5.5 When params are validated: per node, eager or lazy

**Validation belongs to the node.** A node whose params are invalid only matters
if that node runs. For example, a node in a branch arm that no row takes
shouldn't fail the run.

**Cache.** Validation results are cached per `(params document, node path)`.
The params document is identified by content hash, or by the object passed in.
- Each node's generated pydantic model validates that node's view: its local
  params plus the shared keys it uses. The result is a **status**: `UNKNOWN`,
  `OK` or `INVALID` with its errors.
- A new params document starts with every node `UNKNOWN`. Nothing is
  revalidated for a document that was seen before.
- A node's params bundle (the namedtuple passed to its kernel) is built when
  that node is validated. Until then, the prebuilt bundle of defaults is passed,
  which has the same type, so this never recompiles.

**Modes:**

```python
Engine(params_validation="eager")    # default
Engine(params_validation="lazy")
```

| Mode | When a new params document arrives | When a node runs |
|---|---|---|
| `eager` | validate **every** node; any `INVALID` raises immediately (fail fast) | nothing extra |
| `lazy` | nothing | if `UNKNOWN`, validate **this node now** and cache the result; then, if `INVALID`, apply the node's `on_invalid` policy |

With 1000 nodes in lazy mode, the first record validates only the nodes it
actually reaches, perhaps 100. The next record validates only the nodes it
reaches that haven't been seen yet. After a while, every reachable node is
cached, and the per-row cost is one status check per node that runs.

**How "validate on first hit" works in each mode:**
- **`interpreted` and `stepped`** (a Python driver): check the status before
  calling the node, and validate inline. This is trivial.
- **`fused`** (compiled kernel): pydantic can't run inside a kernel, so the
  kernel **suspends**:
  - it takes a status array as an argument;
  - on reaching an `UNKNOWN` node at row *i*, it returns `(node_id, i)`;
  - the driver validates that node, updates the status array and the node's
    params bundle, and relaunches the kernel from row *i*.

  Each first hit costs one kernel re-entry, and once warm there is none. This is
  a requirement on the compiler (T3.2/T3.3).
- **Fallback:** if suspending turns out too costly to build in the fused kernel,
  `lazy` in fused mode falls back to validating every node when the params
  document arrives. The behaviour is the same (only nodes that run can fail);
  just the up-front cost differs. The implementer records which approach was
  taken in `notes/`.

**Per-param policy for invalid values:**
`param(..., on_invalid="error" | "warn" | "default")`.
- `"error"` (the default) raises, naming the node path, the param and the row
  count.
- `"warn"` records a warning and uses the default.
- `"default"` uses the default silently.

A param with no default (`param(required=True)`) that is missing from the
document is `INVALID`.

**Run report.** In lazy mode, the report lists which nodes were validated during
the run, and any invalid ones.

---

## 6. Names, paths and sources

- **Path:** the names of named steps from the root, joined with `/`. Anonymous
  flows are transparent. A `ConfigurableStep`'s nodes sit under its `name`. A
  position inside a `row` node is appended as `#locator`.
- **Paths are unique within one IR**; `to_ir` checks this, with a did-you-mean
  error. The path is the node's id.
- **Source:** the **import path** of the function or class that produced the
  node, for example `credit.rules:cap_by_income` or
  `decider.steps.trees:TreeConfig`. All step code is importable, including
  extensions, so this is stable, readable and serialisable. It answers "which
  code produced this node" even when names repeat across pipelines.

---

## 7. Worked example

```python
import polars as pl
from decider import step, dag, flow, branch, frame_step, param, missing_as
from decider.steps.trees import TreeConfig

def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses

def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment

def affordable(ratio: float, min_ratio: float = param(0.3, shared_key="min_ratio")) -> bool:
    return ratio >= min_ratio

affordability = dag(disposable_income, ratio, affordable, name="affordability")

def term_cap(requested_term: float, ceiling: float = param(60.0, ge=6, le=84)) -> float:
    return min(requested_term, ceiling)

@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float = missing_as(0.0),
                  cap: float = param(48.0, ge=6, le=60)) -> float:
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap

@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)

@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)

def is_private(sector_code: int) -> bool:
    return sector_code == 1

@step(outputs=("band", "band_score"))
def banding(ratio: float) -> tuple[int, float]:
    return (1, 10.0) if ratio > 2 else (0, 0.0)

term = flow(
    term_cap,
    cap_by_income,
    branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
    name="term",
)

@frame_step(reads=["client_id"], writes=["bureau_score"])
def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(BUREAU, on="client_id", how="left")

risk_tree = TreeConfig.load("trees/risk.json")        # name "risk_tree" in the document

pipeline = (join_bureau | affordability | banding | term | risk_tree).emit("term_cap@*")
```

**Authoring tree:** `pipeline.walk()` shows what you wrote.

```
join_bureau                 FrameStep
affordability               DagStep
affordability/disposable_income   FunctionStep
affordability/ratio         FunctionStep
affordability/affordable    FunctionStep
banding                     FunctionStep   outputs=band, band_score
term                        SequentialStep
term/term_cap               FunctionStep
term/cap_by_income          FunctionStep
term/by_sector              BranchStep
risk_tree                   TreeConfig
```

**IR:** `engine.to_ir(pipeline)` gives what runs, with each node's origin.

```
SequenceNode  <root>
  CallNode[frame]   join_bureau                   source=app.pipeline:join_bureau
  SequenceNode      affordability                 source=decider.steps:DagStep
    CallNode        affordability/disposable_income   source=app.pipeline:disposable_income
    CallNode        affordability/ratio           source=app.pipeline:ratio
    CallNode        affordability/affordable      source=app.pipeline:affordable
  CallNode          banding                       source=app.pipeline:banding
  SequenceNode      term                          source=decider.steps:SequentialStep
    CallNode        term/term_cap                 source=app.pipeline:term_cap
    CallNode        term/cap_by_income            source=app.pipeline:cap_by_income
    BranchNode      term/by_sector                source=decider.steps:BranchStep
      CallNode      term/by_sector/is_private
      CallNode      term/by_sector/cap_private
      CallNode      term/by_sector/cap_public
  CallNode[row]     risk_tree                     source=decider.steps.trees:TreeConfig  reference=yes
```

**Params document:** `pipeline.parameters().defaults()` gives:

```json
{
  "shared": {"min_ratio": 0.3},
  "term": {
    "term_cap":      {"ceiling": 60.0},
    "cap_by_income": {"cap": 48.0},
    "by_sector": {"cap_private": {"cap": 54.0}, "cap_public": {"cap": 60.0}}
  },
  "risk_tree": {"hi_thresh": 0.7}
}
```

**Steps are still plain functions in tests:**

```python
assert cap_by_income(term_cap=60.0, min_net_salary=4000.0) == 48.0
assert banding(ratio=3.0) == (1, 10.0)
```

---

## 8. Development loop

- **`to_ir` is pure Python and cheap**: milliseconds for hundreds of steps.
  - It is cached per `(step object, path)`, so unchanged steps return their
    previous IR.
  - Re-running one notebook cell creates one new step object, so only that
    step's IR is regenerated.
- **Compiled code is cached by content** (bytecode plus types), not by path.
  Editing one step recompiles only its kernel.
- **Params changes never regenerate IR or recompile.** A config change
  regenerates only its own step's IR.
- **Live sessions:** `session.replace(path, new_step)` and
  `session.delete(path)` edit a running session. Both regenerate the IR for that
  subtree and re-run downstream from the change. Steps are pure, so this is
  correct.

---

## 9. What lives where

```
decider/steps/                 # the authoring layer (public API)
  base.py                      # Step, auto-conversion, |
  function.py                  # FunctionStep, step()
  frame.py                     # FrameStep, frame_step()
  dag.py  sequential.py        # DagStep/dag(), SequentialStep/flow()
  branch.py  loop.py
  configurable.py              # ConfigurableStep
  values.py                    # Value, ParamRef, TableValue, TableRef
  trees/  tables/  scorecard/  # built-in ConfigurableSteps (later tasks)
decider/engine/ir/             # the IR (engine-facing)
  nodes.py                     # IRNode, CallNode, SequenceNode, BranchNode, LoopNode
  origin.py                    # Origin
  context.py                   # IRContext, engine.to_ir entry and its checks
  decls.py                     # Input, Output, ParamDecl
decider/engine/params/         # harvesting, param(), missing_as(), models, parameters()
```

---

## 10. Acceptance tests for T1.1

1. `step` in every spelling produces identical fields. A decorated step is
   callable with its param defaults.
2. `step(outputs=("a", "b"))` with `-> tuple[float, bool]` works. A mismatched
   annotation length raises when the step is built.
3. `dag(f)` returns `f`'s step. `dag(f, g)` sorts by dependencies. `dag` with two
   writers of one name raises, suggesting `flow`. `f | g` raises, suggesting
   `step(f) | g` or `flow(f, g)`. `f | step` works.
4. `a | b | c` is one flow. `named | c` nests.
5. `pipeline.walk()` and `engine.to_ir(pipeline)` match §7 exactly. That test
   uses a stub `ConfigurableStep` that emits a `row` `CallNode`, because trees
   arrive in T4.
6. Every IR node has an origin with a unique path and an import-path source.
   The same step placed twice gets two distinct paths.
7. Shared params: a type conflict raises the §5.2 message. Differing defaults
   are allowed. `param(on_invalid=...)` is recorded on the `ParamDecl`; running
   the eager and lazy modes is tested in T1.4.
8. `parameters()` and `.defaults()` match §7. A `TableRef` reports its schema.
9. `ConfigurableStep` round-trips through JSON, is frozen, can't be instantiated
   without `to_ir`, and composes with `|`. Type tags:
   - a class with no alias loads by import path;
   - a class with an alias loads by either and dumps the alias;
   - an import path to a non-`ConfigurableStep` is rejected;
   - a duplicate alias raises.
10. `to_ir` is cached: calling it twice on an unchanged step returns the same IR
    object.
11. `decider/steps/` and `decider/engine/ir/` import neither numba nor polars.

---

## 11. How the draft-2 feedback was handled

| Your comment | Outcome |
|---|---|
| Nothing should be called "module"; `IRNode` as the IR base; steps create IR | Adopted: `Step` family (authoring) and `IRNode` family (IR), §2 |
| The config macro means Steps → initial IR → expanded IR → executable | Adopted your two-stage version: every step emits expanded IR directly from `to_ir`. "Recreating" is a new config plus `to_ir` again (§3.6) |
| `shared=` isn't descriptive | Renamed `shared_key=` |
| Global type check for shared params; validation per function | §5.2: one global type table; per-step pydantic models own validation and defaults |
| Every step reports its required params; tables carry a schema | §5.4 `parameters()`, implemented once over the IR; table params carry their schema |
| Params passed as inputs with no rebuild; new rules are out of scope | §4.2 (params are always arguments) and §3.6 (a new rule means reload from config) |
| Import paths as ids | §6: `Origin.source` is the import path. See open question M for config `type` tags |
| `replace` and `delete` on sessions | §8; T6.5 updated |
| Multi-output steps with tuple type hints | §3.2 |
| Param reference format; Haiku usability check midway | Format kept, easy to change. A Haiku fresh-eyes review is added to the plan after T2.2 |
| `flow` / `dag` | Adopted: `dag()` replaces `module()` |

---

### Draft-3 answers

| Question | Decision |
|---|---|
| L. Class names | `SequentialStep` for `\|` / `flow(...)`, and `DagStep` for `dag(...)` |
| M. Type tags | The import path is the default. A declared short alias also registers the class (§3.6) |
| N. Shared defaults that disagree | No global check. Validation is per node, with an engine-level `eager`/`lazy` mode (§5.5) |

---

## 12. Open questions

None. IR is signed off. Lazy validation on first hit (§5.5) is the target;
validating every node up front is the accepted fallback in fused mode.
