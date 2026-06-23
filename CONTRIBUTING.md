# Contributing

We value and encourage community contributions. To get started, please follow these guidelines:

1. [Code of Conduct](#1-code-of-conduct)
2. [Issues](#2-issues)
3. [Vulnerabilities](#3-vulnerabilities)
4. [Development](#4-development)
5. [Pull Requests](#5-pull-requests)

## 1. Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 2. Issues

Engagement starts with an Issue where conversations and debates can occur around [bugs](#bugs) and [feature requests](#feature-requests):

- ✅ **Do** search for a similar or existing Issue prior to submitting a new one.
- ❌ **Do not** use Issues for personal support. Use [Discussions](https://github.com/capitec/dsp-decision-engine/discussions) or [StackOverflow](https://stackoverflow.com/) instead.
- ❌ **Do not** side-track or derail Issue threads. Stick to the topic, please.
- ❌ **Do not** post comments using just "+1", "++" or "👍". Use [Reactions](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) instead.

<h3 id="bugs">👾 Bugs</h3>

- ✅ **Do** search for a similar or existing Issue prior to submitting a new one.
- ✅ **Do** describe the bug concisely. **Avoid** adding extraneous code, logs, or screenshots.
- ✅ **Do** attach a minimal test or example to demonstrate the bug.

<h3 id="feature-requests">💡 Feature Requests</h3>

- ✅ **Do** search for a similar or existing Issue prior to submitting a new one.
- ✅ **Do** provide sufficient motivation and use case(s) for the feature.
- ❌ **Do not** submit multiple unrelated requests within one request.

> **TIP:** Engage as much as possible within an Issue before proceeding with contributions.

## 3. Vulnerabilities

- ✅ **Do** refer to our [Security Policy](https://github.com/capitec/dsp-decision-engine/security/policy) for more information.
- ✅ **Do** report vulnerabilities via this [link](https://github.com/capitec/ml-decision-engine/security/advisories/new).
- ❌ **Do not** open a public Issue or Discussion for security vulnerabilities.

## 4. Development

<h3 id="branches">🌱 Branches</h3>

- `feature/*` — feature development; PR target is `main`.
- `main` — stable branch; tagged releases are cut from here.

<h3 id="dependencies">🔒 Dependencies</h3>

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) for environment and dependency management.

<h3 id="project-setup">📦 Project Setup</h3>

```bash
# 1. Clone and enter the repo
git clone https://github.com/capitec/dsp-decision-engine.git
cd dsp-decision-engine

# 2. Install all dependencies (creates .venv automatically)
uv sync --all-extras

# 3. Run the tests
uv run pytest
```

If you are behind a corporate proxy that uses a private CA, `uv` is already
configured to use the system certificate store (`system-certs = true` in
`pyproject.toml`).

<h3 id="directory-structure">📂 Directory Structure</h3>

```
decider/                  Core library
  cli/                    `decider` CLI (click)
  config/                 Versioned config management
  modules/                Module primitives (expression, join, sequential, union)
  serving/                HTTP servers (Starlette, Sanic)
  templates/              Scaffolding templates and scaffold.py renderer
  magics/                 Jupyter %%module magic
docs/examples/            Example notebooks
projects/                 End-to-end project examples
tests/                    pytest suite
pyproject.toml            Project metadata, dependencies, tool config
uv.lock                   Locked dependency graph (committed)
```

<h3 id="naming-conventions">🏷 Naming Conventions</h3>

- ✅ **Do** follow PEP 8.
- ✅ **Do** name classes in `CamelCase` and functions/modules in `snake_case`.

<h3 id="code-quality">🔍 Code Quality</h3>

- ✅ **Do** adhere to PEP 8 style guidelines.
- ✅ **Do** use `ruff` or `black` for formatting before opening a PR.

<h3 id="testing">🧪 Testing</h3>

- ✅ **Do** write tests under `tests/` using `pytest`.
- ✅ **Do** ensure all tests pass before submitting a Pull Request (`uv run pytest`).

## 5. Pull Requests

- ✅ **Do** ensure your branch is up to date with `main`.
- ✅ **Do** ensure there are no merge conflicts.
- ✅ **Do** make sure all tests pass.
- ✅ **Do** provide a clear description of the changes and their purpose.

> **TIP:** Review the existing codebase and follow the conventions used throughout the project.

---

Thank you for contributing! We appreciate your efforts to improve the project.
