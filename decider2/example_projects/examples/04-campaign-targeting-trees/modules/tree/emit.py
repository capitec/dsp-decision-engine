"""Codegen for one tree shape.

What is emitted, for a 74-node tree, is roughly 300 lines of nested real `if`
statements over scalars, with shared successors emitted once as `@njit` helpers
so the line count is O(nodes) and not O(routes).  Node 6 in campaign 23 is
reached from node 3's false arm and node 4's true arm; it is emitted once and
called twice.

Three properties of the emitted source, each load-bearing:

  1.  **No literal is an immediate.**  Every threshold reads `thr[17]` from the
      thresholds array.  This is the mechanism behind the whole overlay story: a
      volume dial substitutes an array element at run time and the compiled code
      is untouched, so §5.3.4 req 1 ("an overlay is not an edit... must not change
      any node_key") is true by construction rather than by discipline.  It also
      means the 30 threshold moves a month cost zero compilation.

  2.  **The route folds as it walks.**  Two integer ops per node, an immediate
      XOR of the edge id and a multiply.  Not a tap, not a trace mode, not
      something a run can be optimised into dropping — the emitted traversal
      cannot produce a leaf without producing a route (spec §5.4's entire reason
      for being a separate stage).

  3.  **Deterministic.**  Node emission order is the canonical topological order
      with (level, node_key) as the tie-break, so the generated source is
      byte-identical between the publication job that compiled it and the cycle
      that loads it.  Doc 08 §4.2's warning about an unstable topological sort
      invalidating the numba cache is the failure this tie-break prevents.

The emitted shape, illustrated on campaign 23 v11:

    @njit(cache=True)
    def _n_a1f45e9c2b70d863(f, thr, route):          # "node 6"
        route = _edge(route, 0x8c11d0a47b3e5926)     # placeholder; see below
        held = (f.discretionary_income >= thr[11]
                and f.estimated_instalment_to_income <= thr[12]
                and (f.employment_type_code == 1
                     or f.employment_type_code == 2
                     or f.employment_type_code == 4))
        if held:
            return _n_e83017bd6f2c94a5(f, thr, _edge(route, 0x3f19c72ba0d4e851))
        return _leaf(903, _edge(route, 0x77ba03e5c1948d2f))

Small-set membership compiles to an `or` chain up to 8 codes and to a bitmask
test above it; both are register-resident and neither allocates.
"""

from __future__ import annotations

from .document import TreeDocument


def emit_tree_kernel(doc: TreeDocument, *, input_schema, parallel: bool = True) -> str:
    """Return the generated module source for this tree shape.

    `parallel=True` is the default here, unlike doc 02 §3.3's serial default.
    Campaign trees are the one workload where `prange` is unambiguously right:
    the row loop is embarrassingly parallel, the per-row cost is a *distribution*
    of 2..12 node visits with no cross-row dependence, and the batch is 6.7 M
    rows on average.  It is still authored, not inferred — `parallel(...)` in the
    pipeline expression — and the flag is part of the shape fingerprint so the
    audit record says which variant ran (doc 08 §8).
    """
    pass  # topological emit, helpers for shared successors, thr[] reads, route fold


def emit_portable_interpreter(doc: TreeDocument) -> str:
    """Emit the dependency-free Python evaluator that ships **with the artefact**.

    Spec §9.1 requires a non-engineer to render a path from stored evidence with
    the decision system unavailable, and §5.4.2(4) requires an analyst to
    re-derive a leaf on a laptop.  Neither can import decider2, numba or polars.

    So the artefact carries a ~200-line pure-Python interpreter over the closed
    condition algebra.  This is not a violation of doc 08 §1.3's "do not build a
    general interpreter": that ruling is about the *hot path*, where an
    interpreter would be the most complex code in the framework and exist solely
    to dodge a compile.  Here the interpreter is the reference semantics, it runs
    on one record at a time in a spreadsheet-adjacent context, and its agreement
    with the kernel is an automated test — the equivalence ladder with a fourth
    rung (`portable == interpreted == stepped == fused`).  See DEMANDS #12.
    """
    pass  # emit evaluator + the canonical conditions as data
