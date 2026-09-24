# {{name}}

- `{{name}}/pipeline.py`: the steps, and `build()`, which returns the pipeline.
- `{{name}}/inference.py`: the `Handler` that serves it; override its `*_fn` methods to change request handling.
- `configs/<version>/`: one directory per config version; `params.json` is the params document.
- `sample_request.json`: one request; `decider build` warms the kernels with it.
- `.env`: the `DECIDER_*` settings `decider build` and `decider serve` read.

```bash
pytest -q
decider build
decider serve
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
```

`decider guide` explains the rest.
