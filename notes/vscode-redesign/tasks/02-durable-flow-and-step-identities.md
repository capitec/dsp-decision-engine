# 02 — Add durable flow and step identities

**Depends on:** 01  
**Blocks:** 04, 05, 07, 08, 09, 10, 11

**Experiment gate:** `experimentation/05-step-id-findings.md` establishes the
candidate syntax; `experimentation/05-step-id-follow-up.md` must validate it
against the real engine and a real source rewrite before the interface freezes.

## Outcome

Every production pipeline flow and step has a stable identity suitable for trace
decoding, comparison, source navigation, parameter references, debugging, and
experiment assets, without materially reducing source readability.

## Work

- Apply the validated finding: optional `id=` on constructors/decorators,
  `secrets.token_hex(6)` opaque IDs, and JSON-config `id` fields. Retain the
  original syntax spike and complete its follow-up integration proof before
  implementing the generator.
- Design the identity model: derived IDs for discovery and explicit committed
  flow/step IDs for durable references.
- Choose the least noisy source syntax that fits the existing step declarations.
- Implement a dedicated ID-generation command that updates source only when all
  affected files are tracked and the working tree is clean.
- Make the command explain failures and the remediation steps instead of
  partially changing source.
- Preserve existing source-to-step mapping and add migration/compatibility
  tests for refactors, extraction, reordering, and user-supplied IDs.
- Determine how configurable data assets and parameters refer to stable steps.
- Require persisted references to capture descriptive metadata—flow/step names
  and source locations—as they existed at capture time, so unresolved IDs can
  render a repairable partial result after a refactor.

## Important decisions

- Derived IDs cannot promise stability across arbitrary refactors; they must
  never be presented as audit-grade durable identities.
- Generated source changes are deliberate and reviewable, not background
  editor actions.
- Explicit IDs may be optional during development but are required where traces,
  comparisons, or experiments need durable links.

## Done when

- The generator is safe, produces readable code, and its actual rewrites are
  validated against formatting, comments, source mapping, and a Git fixture.
- Committed flow and step IDs survive the expected code evolution scenarios.
- All downstream consumers can resolve the same ID to flow structure and source.
