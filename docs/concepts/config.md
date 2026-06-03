# Config & Versioning

Decider modules are serialised to and from JSON configs, enabling versioned, auditable deployments.

## Loading a module from config

Every module exposes `.load(config_dict)` which returns a bound, runnable instance:

```python
config = {
    "type": "Scorer",
    "scorer_config": {"threshold": 4.5},
}
module = GraphModule.load(config)
result = module.run(df)
```

## Saving a config

```python
config_dict = module.model_dump()
import json
json.dumps(config_dict, indent=2)
```

## Versioning with git tags

Configs stored on disk are snapshots. Pair them with a git tag to pin the exact code version:

```bash
git tag v1.2.0
# store configs/v1.2.0/loan_decision.json alongside the tag
```

## Config directory layout

The default layout expected by `decider serve`:

```
project/
├── configs/
│   └── loan_decision.json
├── extensions/
│   └── my_scorer/
│       └── __init__.py
└── generate.py
```

## Environment-specific overrides

Use Pydantic `BaseSettings` inside your config models to pull values from environment variables at load time, keeping secrets out of version-controlled JSON files.
