# 02 — Add durable flow and step identities

**Depends on:** 01  
**Blocks:** 04, 05, 07, 08, 09, 10, 11

## Outcome

Every production pipeline step has a stable identity suitable for trace
decoding, comparison, source navigation, parameter references, debugging, and
experiment assets, without materially reducing source readability.

## Work

- Run `experimentation/05-step-id-source-syntax-spike.md` before selecting the
  generated source syntax or implementing the generator.
- Design the identity model: derived IDs for discovery and explicit committed
  IDs for durable references.
- Choose the least noisy source syntax that fits the existing step declarations.
- Implement a dedicated ID-generation command that updates source only when all
  affected files are tracked and the working tree is clean.
- Make the command explain failures and the remediation steps instead of
  partially changing source.
- Preserve existing source-to-step mapping and add migration/compatibility
  tests for refactors, extraction, reordering, and user-supplied IDs.
- Determine how configurable data assets and parameters refer to stable steps.

## Important decisions

- Derived IDs cannot promise stability across arbitrary refactors; they must
  never be presented as audit-grade durable identities.
- Generated source changes are deliberate and reviewable, not background
  editor actions.
- Explicit IDs may be optional during development but are required where traces,
  comparisons, or experiments need durable links.

## Done when

- The generator is safe, deterministic, and produces readable code.
- A committed ID survives the expected code evolution scenarios.
- All downstream consumers can resolve the same ID to flow structure and source.
