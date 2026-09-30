# Product feedback received

## Graph scale budget

The representative workload includes small flows and very large flows with:

- thousands of nodes;
- dense, highly connected edges; and
- approximately 6–10 nesting levels.

`example_projects/10-retail-credit-e2e/sonnet/pipeline.py` is a representative
source shape: composed phase units, relabelled data dependencies, nested flows,
and loop logic. The graph spike should use it and its expanded IR as a fixture,
then add generated dense graphs when it does not reach the target size.

Exact counts and response-time targets are not available. The spike must first
measure the current baseline and propose concrete budgets for review; it must
not select implementation limits from guessed numbers.
