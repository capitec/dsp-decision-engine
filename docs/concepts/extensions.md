# Extensions

Extensions are Python packages or modules that live in an `extensions/` directory alongside your project. Decider discovers and imports them automatically at startup, making their module types available in the registry.

## Inline extension (single file)

The simplest form — a single `__init__.py` inside `extensions/<name>/`:

```
extensions/
└── my_scorer/
    └── __init__.py   # defines and registers MyScorer
```

## Package extension (shareable)

For extensions you want to share or version independently, use a `uv` src-layout package:

```
extensions/
└── my_pkg/
    ├── pyproject.toml
    └── src/
        └── my_pkg/
            ├── __init__.py
            └── my_scorer.py
```

Decider discovers both layouts automatically.

## Jupyter magic

In a notebook, use the `%%module` magic to create or update an extension inline:

```python
%load_ext decider.magics
```

```
%%module MyScorer

def score(income: float, debt: float) -> float:
    return income / (debt + 1)
```

This writes the module file to `extensions/my_scorer/__init__.py`, imports it, and registers it — all in one cell. Re-running the cell reloads and re-registers without duplicating the discriminator.

### Package mode

```
%%module MyScorer --package my_pkg
```

Creates `extensions/my_pkg/src/my_pkg/my_scorer.py` as a proper installable package.

## CLI scaffolding

```bash
# scaffold a standalone module
decider template module MyScorer

# scaffold inside a package
decider template module MyScorer --package my_pkg

# scaffold a full project
decider template project my_project
```
