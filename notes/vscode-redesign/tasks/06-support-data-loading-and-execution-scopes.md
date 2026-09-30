# 06 — Support data loading and explicit execution scopes

**Depends on:** 01, 01a, 03

**Blocks:** 07, 09, 10, 11, 12

## Outcome

Let users load supported data, find records, and choose a semantically correct
execution scope before tracing or debugging a flow.

## Work

- Define and implement JSON, CSV, and Parquet ingestion suitable for
  Redshift-export workflows.
- Surface schema, row count, record identifiers, load failures, missing
  required values, and unavailable defaults before a run starts.
- Add record search/filtering and a stable focused-record representation.
- Offer the existing ID-like heuristic as a load-time suggestion, support an
  explicit ID column or composite key, and reject duplicate/missing durable
  identity when an experiment or debugger reproduction needs one.
- Define **selected record** and **whole frame** execution modes. Detect
  frame-step requirements and refuse or redirect invalid record-only execution.
- Define hybrid inspection as full-frame execution with record-scoped display,
  tracing, and breakpoints. Defer dependency-closure execution until it can
  prove identical semantics.
- Add pre-flight validation for data compatibility, parameter constraints,
  version capabilities, and an initial run-cost estimate before execution.
- Integrate Polars where it provides local scalable data execution without
  adding remote-resource provisioning.

## Important decisions

- The framework's missing-value semantics remain authoritative; the tooling
  reports incompatibility rather than inventing values.
- Whole-frame semantics are required where a frame calculation depends on
  aggregates or other rows.
- Optimise practical local execution; clients must sample when data exceeds
  their available resources.

## Done when

- Users can load, validate, find, and select data without constructing debug
  launch JSON manually.
- The execution scope is visible and semantically safe.
- Frame-sensitive steps never silently run against an insufficient subset.
