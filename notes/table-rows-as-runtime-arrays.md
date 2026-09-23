# Table rows, tree data and thresholds are runtime arrays

**Decision:**
- A decision table's rows, a tree's thresholds and a rule's thresholds and
  enabled flags reach the kernel as runtime arrays or params arguments. They are
  never literals in generated source.
- Editing a value or a row never recompiles; only a schema change (column set,
  dtypes) does.

**Why:**
- **Table compile cost doesn't depend on row count.** The same 31-line source
  (sha `93277fad`) was emitted for 2, 50 and 1000 bands, so every table after the
  first is a cache hit.
  - The cost moves to answer time instead: a table is a linear scan, 70 ns/row at
    2 bands and 1045 ns/row at 1000.
- **Thresholds as arguments:**
  - 8 retunes as literals meant 8 full recompiles, 343 ms each at 5 rules. As
    arguments: 0 compile events, and the signature count stayed at 1.
  - Runtime cost: +1–4.5 ns/row (+3.1% at 10 rules, +11.2% at 30). That is
    nothing against a 20 ms single-record budget.
- **Rule enablement as a mask array:** −2.0% at 10 rules (noise) and +5.4% at
  30. Break-even against one literal-form recompile is about 427M rows.
- **No stale constants.** A retune writes no source, so the stale-constant cache
  bug can't happen on the path a business user edits (see
  `numba-compile-cache.md`).

**What we tried:**
- Hoisting thresholds was also expected to cut compile time. It doesn't: emitted
  lines were identical (11/25/65 at 3/10/30 rules), and the compile-time change
  had no consistent sign. The case rests only on "no recompile on retune".
- **Codegen'd trees vs an interpreted array walk:**
  - Codegen is 2.0–2.7× faster per row through depth 7.
  - Codegen then becomes unpredictable. At depth 10 it measured 134–242 ns/row
    across identical reruns, an 80% spread, where the interpreted walk stayed
    under 2%. It also compiles in 20 s.
  - Codegen hits CPython's 100-level indentation limit, found with a 128-node
    one-sided chain.
  - The data-walked kernel has no line cap, no indentation limit and no
    per-shape compile.
- **Compile cost of generated source** grows faster than linearly with emitted
  lines: lines^1.4 for rulesets, 56.6 s at 100 rules. A cap of about 500 lines is
  roughly 30 rules.

**Source:** `decider2/docs/EXPERIMENTS.md` (§G, §L, §M, §P, §Q);
`decider2/docs/08-configuration-and-lifecycle.md` (§2, §2.1);
`experimentation/rule-thresholds-as-args/`;
`experimentation/tree-codegen-vs-interpreted/`; `IR.md` (§5.3)
