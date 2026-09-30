# Kimi K3 feedback — v1 (original, pre-answers)

Preserved for consolidation. Superseded by `feedback.md` (v2), which reviews
the answers in `questions.md`. The original question text and my leanings are
also preserved in `questions.md`.

## Verdict (v1)

The component separation is sound: three user modes, one shared flow/source
model, semantics pushed down into `decider` (tracing, checks, experiments),
extension and notebooks as adapters, small MCP surface. The risk was a set of
cross-cutting semantics no task owned: identity/provenance semantics, trace
ordering and nogil capture risk, trace-vs-debugger division of labour, hybrid
execution scope, numeric equality, nondeterminism, cancellation, contract
versioning, the unstated UI modal model, MCP topology, orphaned stories 4/5,
and assorted smaller items (graph scale budgets, notebook-adapter proof,
extension↔decider version skew, naming, engine targeting, flow composition).

All of these were raised as questions A1–G3 in `questions.md` and answered.
See `feedback.md` for what the answers settle and what they newly imply.
