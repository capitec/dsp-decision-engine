# 08 — Build generic checks and diagnostics

**Depends on:** 01, 02  
**Blocks:** 11, 12

## Outcome

Provide a Python-level check interface that helps clients detect common flow
quality and correctness risks and lets CI, notebooks, and VS Code consume one
structured report format.

## Work

- Design the check module interface, suite registration, result/report model,
  severity conventions, source links, and machine-readable output.
- Implement a default suite of static diagnostics, beginning with wall-clock
  reads, non-pure/internal-state behaviour where detectable, and numeric
  overflow/underflow or floating-point-sensitivity risks.
- Assess and, where justified, prototype whole-path checks and revision
  comparison: structural equivalence where proven and generated
  threshold/regime cases otherwise.
- Create a supported client extension seam for custom checks without making
  custom checks a dependency of the default suite.
- Expose reports in VS Code as navigation and explanation surfaces, and make
  CLI/Python invocation suitable for client CI policy.

## Important decisions

- `decider` reports findings; clients decide warning versus release-blocking
  policy.
- Start generic. Regulated-domain classifications and centrally governed rule
  ownership are explicitly out of initial scope.
- Soundness claims for path checking or equivalence must be precisely scoped;
  do not imply proof when the analysis is heuristic.

## Done when

- `decider.check.run(pipeline, suites=[decider.checks.default_suite])` returns
  a documented structured report.
- The default suite identifies representative known defects with source-aware
  findings.
- The same report is usable from CI and navigable in VS Code.

