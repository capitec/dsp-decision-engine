# Pipeline structure lives in Python

**Decision:** Which steps exist, how they are wired and in what order is Python
code. Config carries params documents and `ConfigurableStep` documents (tree
bodies, table rows, scorecards), and it references code only by registered tag.
Config-defined structure stays possible later, as configurable forms of the
combining steps, without a redesign.

**Why:** decider_old had structure in config, and it produced two sources of
truth that drifted apart:
- Execution order was kept by hand in JSON with typed UUIDs, and it was known to
  be wrong: `main.json` had 51 steps, the generated `flow.json` had 120, and the
  real C# flow had 192. This was recorded as "Unresolved".
- Elsewhere, order was stated only in a comment.
- Values were duplicated between config and code: `"max_term": 84` in a config
  file, with `84.0` also hardcoded in two modules. Editing the JSON might not
  change behaviour.
- `|` ran stages in list order, not as a DAG. One logical module was split into
  nine registered stage modules, and 21 of 22 functions were identity aliases.
- The graph couldn't be drawn without building it, and building it needed valid
  params for every registered type. Nested validation errors also lost the step
  index.

When structure is code, it is reviewed, versioned by commit, and its order is
derived from the graph rather than maintained by hand.

**What we tried:**
- **`output_fn` as a `{module_name, function_name}` pointer in config.** It gave
  up config as inspectable data and was `null` in every production config.
- **decider2's three document kinds:**
  - params: free to change;
  - interiors: recompile, interface fixed;
  - structure: `Admit.COMPOSITION`, never shipped.

  Two rules came out of this: "config may reference code by registered id, never
  contain code or an unregistered pointer", and "prefer generating Python over
  widening what config admits". Both carry over.

**Source:** `decider2/docs/01-motivation-and-evidence.md` (§5.2–§5.4, §5.7);
`decider2/docs/08-configuration-and-lifecycle.md` (§1, §2, §7);
`decider2/docs/03-authoring-api.md` (document kinds table). In these docs,
"`decider`" means what is now `decider_old/`.
