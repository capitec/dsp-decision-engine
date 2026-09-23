# Tree documents: two formats, one `Tree`

`decider.steps.trees` accepts decider_old's v3 tree documents and its flat-rule
documents (`flat_rule`, `prioritized_flat_rule`). Each validates into the same
`Tree`, so the compiler never sees which format a tree came from.

- **The document is what's stored; `Tree` is derived.** `TreeConfig` holds a
  `TreeDocument` so a v3 tree keeps its node positions for the editor on
  round-trip. `to_tree()` returns the `Tree` built at validation time, so
  structural errors surface when the document loads.
- **`Tree` is a graph, not nested rules.** `nodes: {id: Node(data, children)}`
  with `children[i]` the node id branch `i` leads to (`None` = default row),
  plus `rules` (root ids in priority order) and `mode`. A v3 DAG stays a DAG; a
  flat document's shared branch (two cases pointing at one `branches[j]`)
  becomes one node. Only nodes reachable from a root are kept, which is what
  decider_old executed.
- **Flat nodes without an id get a path id** (`"0"`, `"0.1"`, `"0.1.2"`), not
  decider_old's `uuid4`, so the same document always gives the same ids for
  breakpoints and cache keys. The same id twice is fine only when both nodes are
  identical (decider_old's v3-to-flat conversion duplicated shared subtrees
  under one id).
- **Version detection** follows decider_old (`formatVersion`, then
  `format_version`, dict-shaped `nodes` = v0), plus a `"v<n>-tree"` `type`. A
  document with none of these is read as v3, where decider_old guessed v1: a
  real v1 document always carries `formatVersion: 1` or `type: "v1-tree"`.
  v0, v1 and v2 raise a deprecation error naming the version.
- **`{"key": k}` (decider_old's `InputRef`) is read as `ParamRef(param=k)`.**
  A document's `parameters` block (`{k: {type, default_value}}`) fills in the
  default of every local ref to `k` that has none, so decider_old documents keep
  their defaults.
- **Thresholds are `Value[int | float]`, not `Value[float]`.** A float would
  round an int64 id above 2**53 before the kernel sees it.
- **An empty composite is rejected**, as decider2 did; decider_old evaluated an
  empty AND as false, which nobody should rely on. Empty `Cases*` nodes are
  accepted and always take `otherwise`.
- **Dropped:** `is_null`/`is_not_null` (null handling is per input),
  `null_handling` on string matches (ignored), and `output_fn`,
  `post_process_fn`, `format_prioritized_fn` (Python function references;
  rejected by name). `input_schema`, `parameters_col`, `keep_input` and
  `use_optimized_execution` are ignored.
