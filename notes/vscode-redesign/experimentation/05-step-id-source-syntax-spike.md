# Experiment 05 — Durable step-ID source syntax

**Run during:** task 02, before the ID generator is implemented  
**Feeds:** tasks 02, 04, 08, 09, 10, 11

## Question

What explicit step-ID syntax is readable, preserves existing flow semantics,
and survives normal source evolution without creating needless merge friction?

## Method

- Prototype generated opaque IDs in the existing step/decorator forms.
- Test rename, extraction, reordering, copy/paste, independent branch edits,
  duplicate detection, source formatting, source mapping, and clean-tree
  generation failure paths.
- Compare resulting pipeline readability with an ID-free baseline.

## Decision outputs

- Exact generated source syntax and ID format.
- Readability rule and generator safety/error behaviour.
- Duplicate/merge handling and migration guidance.

