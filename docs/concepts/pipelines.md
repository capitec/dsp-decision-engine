# Pipelines

Modules compose into pipelines using `|` (sequential) and `&` (join) operators.

## Sequential pipelines (`|`)

`A | B` passes A's output as input to B. Use this when each stage enriches the data for the next.

```python
from decider import generate_from_functions, SequentialModule

Enricher = generate_from_functions(enrich_fn, name="Enricher")
Scorer = generate_from_functions(score_fn, risk_fn, name="Scorer")

Pipeline = Enricher | Scorer
result = Pipeline.load({}).run(df)
```

## Join pipelines (`&`)

`A & B` runs both modules on the same input and merges their outputs column-wise.

```python
CreditScore = generate_from_functions(credit_fn, name="CreditScore")
AffordabilityScore = generate_from_functions(afford_fn, name="AffordabilityScore")

Combined = CreditScore & AffordabilityScore
```

## Nesting

Pipelines can be nested to arbitrary depth:

```python
pipeline = (StageA | StageB) & StageC | FinalStage
```

## Named pipelines

Wrap any composed pipeline as a `GraphModule` to give it a name and make it JSON-serialisable:

```python
from decider import register_graph_module

@register_graph_module
class LoanDecision(SequentialModule):
    steps = [Enricher, Scorer]
```
