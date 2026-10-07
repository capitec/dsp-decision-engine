# Steps and the IR

!!! note "Draft"
    Outline only.

- Steps are what users write; the IR is what the engine runs
- `Step.to_ir()`: each step produces its own expanded IR
- The node types: `CallNode`, `SequenceNode`, `BranchNode`, `LoopNode`, `ScatterGatherNode`, `FiniteStateMachineNode`
- Declarations: `Input`, `Output`, `ParamDecl`, `NullPolicy`
- Origins: every node records the code that produced it, and origins never enter compile cache keys
- `ConfigurableStep`: trees, tables and scorecards as documents that expand to IR
- Adding a step type: what to implement
