# Experiment 05 follow-up — Durable ID integration proof

**Run before:** task 02 implements or freezes the public ID interface  
**Builds on:** `05-step-id-findings.md`

## Confirmed direction

Use a uniform optional `id="0123abcdef45"` keyword on step constructors and
decorators. IDs are short, opaque 12-hex tokens generated with
`secrets.token_hex(6)`, committed in source, and carried alongside existing
name/path/source identity.

## Remaining validation

The first spike validated syntax with `ast` and a model of evolution. Before
task 02 starts implementation, prove the design against the real engine:

1. Construct and lower every supported declaration shape to IR, including
   `@step`, `@frame_step`, wrapped imported/reused functions, `flow`, `dag`,
   `branch`, `loop`, `each`, `optimise`, and JSON tree/table/scorecard configs.
   Confirm the ID reaches the intended `Origin`/IR reference without changing
   name, path, `step_map`, or execution semantics.
2. Decide and test flow IDs as well as step IDs. The generator must locate the
   pipeline/root-flow declaration and make `flow ID + step ID` a global
   reference.
3. Test reuse/composition: one decorated step used in multiple flows, a
   sub-flow called by a parent, copied configurable assets, and imported steps
   from a package that does not yet have IDs.
4. Prototype the actual source transformation, not only `ast.parse`. It must
   preserve comments, existing decorators, imports, formatting, and source
   mapping; parse-before-write, clean-tree, idempotence, duplicate detection,
   and failure atomicity need a temporary Git fixture.
5. Test generated-ID collision and uniqueness scope. The documented
   repository-wide duplicate scan is stricter than the flow-local reference
   scope; make that intentional and state how external-package IDs participate.

## Decision gate

Task 02 may adopt the syntax and generator only after these real-engine and
real-rewrite checks pass. A failure may change the implementation mechanism,
but not the established invariants: explicit additive IDs, readable source,
and durable identity independent of name/path/order.

