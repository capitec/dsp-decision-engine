# {{name}}

- `pipeline.py`: the steps, and `build()`, which returns the pipeline.
- `inference.py`: the `Handler` that serves it; override its `*_fn` methods to change request handling.
- `configs/<version>/`: one directory per config version; `params.json` is the params document.
- `sample_request.json`: one request; `decikit build` warms the kernels with it.
- `.env`: the `DECIKIT_*` settings `decikit build` and `decikit serve` read.

The project directory itself is the package, so `import {{name}}` resolves to it.

```bash
pytest -q
decikit build
decikit serve
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
```

`decikit guide` explains the rest.
