# Experiment 02 — Flow graph scale and interaction

**Run during:** task 05, before graph/inspector implementation freezes  
**Feeds:** tasks 05, 07, 11

## Question

Which graph-library configuration and interaction contract keep large,
nested flows understandable while preserving selection and viewport context?

## Method

- Collect representative real-flow fixtures, including
  `example_projects/10-retail-credit-e2e/sonnet/pipeline.py` and its expanded
  IR. Cover thousands of nodes, dense edges, 6–10 nesting levels, and
  source-map complexity; add generated dense graphs where the real fixture is
  smaller.
- Test required interactions: pointer-centred zoom, canvas pan, fit-to-flow,
  fit-to-selection, collapse/expand anchored on the selected node, source
  navigation, MCP-driven highlighting, and breakpoint decorations.
- Measure layout time, interaction latency, and webview memory.
- Compare a minimal edge-label approach with selected-step, selected-edge, and
  selected-value progressive disclosure.

## Decision outputs

- Flow-size fixture set and explicit rendering/interaction budgets.
- The graph-library gesture configuration and required fallbacks.
- The viewport/selection preservation algorithm.
- The amount of dependency information visible before selection.
