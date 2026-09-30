# Experiment 03 — Experiment interface and local scale

**Run during:** task 09, before a public experiment interface or YAML schema is frozen  
**Feeds:** tasks 09, 10, 11

## Question

What is the smallest portable experiment interface that supports reproducible
local Polars execution, scenario comparisons, declared overrides, revision
references, aggregation, cancellation, and optional Python hooks?

## Method

- Compare at least two candidate Python/YAML interfaces against plain Python,
  VS Code, and headless MCP callers.
- Reuse and evaluate existing session, fork/sweep, comparison, corpus, and
  equivalence machinery before adding a new execution path.
- Exercise input fingerprint mismatch, symbolic and resolved revisions,
  declared override validation, float tolerance, nondeterministic steps,
  cancellation, per-scenario failure, resume, and manifest generation.
- Benchmark representative local data sizes and scenario combinations with
  Polars; state the practical local limit rather than extrapolating to
  distributed execution.

## Decision outputs

- Chosen deep interface, with rejected alternatives and rationale.
- Versioned `experiment.yaml` and result-manifest shape.
- The portable declarative subset versus optional Python `tests/` and
  `graphs/` hooks.
- Job/progress/cancellation semantics and result ownership.
