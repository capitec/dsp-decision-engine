# {{name}}

- `pipeline.py`: the steps, and `build()`, which returns the pipeline.
- `inference.py`: the `Handler` that serves it; override its `*_fn` methods to change request handling.
- `configs/<version>/`: one directory per config version; `params.json` is the params document.

`decider build` checks and warms the latest config version; `decider serve` serves it.
Score a record with `curl -d '{"income": 1000, "debt": 250}' localhost:8080/invocations`.
