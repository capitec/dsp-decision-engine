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
- ❌ **Do not** use Issues for personal support. Use [Discussions](https://github.com/your-repo/discussions) or [StackOverflow](https://stackoverflow.com/) instead.
- ❌ **Do not** side-track or derail Issue threads. Stick to the topic, please.
- ❌ **Do not** post comments using just "+1", "++" or "👍". Use [Reactions](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) instead.

<h3 id="bugs">👾 Bugs</h3>

A bug is an error, flaw, or fault associated with *any part* of the project:

- ✅ **Do** search for a similar or existing Issue prior to submitting a new one.
- ✅ **Do** describe the bug concisely. **Avoid** adding extraneous code, logs, or screenshots.
- ✅ **Do** attach a minimal test or example to demonstrate the bug.

<h3 id="feature-requests">💡 Feature Requests</h3>

A feature request is an improvement or new capability associated with *any part* of the project:

- ✅ **Do** search for a similar or existing Issue prior to submitting a new one.
- ✅ **Do** provide sufficient motivation and use case(s) for the feature.
- ❌ **Do not** submit multiple unrelated requests within one request.

> **TIP:** Engage as much as possible within an Issue before proceeding with contributions.

## 3. Vulnerabilities

A vulnerability is a security-related risk associated with *any part* of the project or its dependencies:

- ✅ **Do** refer to our [Security Policy](https://github.com/your-repo/security/policy) for more information.
- ✅ **Do** report vulnerabilities via this [link](https://github.com/your-repo/security/advisories/new).
- ❌ **Do not** report any Issues or mention vulnerabilities in public Discussions for discretionary purposes.

## 4. Development

<h3 id="branches">🌱 Branches</h3>

- `develop` - Default branch for all feature development and Pull Requests.
- `main` - Stable branch for all periodic releases.

<h3 id="dependencies">🔒 Dependencies</h3>

* Python (>= 3.8)
* `pip` for package management. Use `pip install -r requirements/all.txt` to install dependencies.
* Optional: Set up your environment using `conda`, `virtualenv`, or another method. Refer to [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) for guidance.

<h3 id="project-setup">📦 Project Setup</h3>

1. [Fork](https://github.com/your-repo/fork) the repository and create a branch from `develop`.
2. Clone the forked repo, checkout your branch, and install the dependencies with `pip install -r requirements/all.txt`.
3. Run tests using `pytest` to ensure everything is working correctly.

<h3 id="directory-structure">📂 Directory Structure</h3>

When contributing, please note the following key files and directories:
├── docker
│ ├── Dockerfile
├── docs
│ ├── index.rst
│ ├── ...
├── requirements
│ ├── all.txt
│ ├── ...
├── spockflow
│ ├── components
│ │ ├── scorecard
│ │ ├── tree
│ │ ├── dtable
│ ├── inference
│ ├── ...
├── core.py
├── exceptions.py
├── nodes.py
├── tests
│ ├── test_example.py
│ ├── ...


* `docker` - Contains all files related to Docker images.
* `docs` - Documentation files.
* `requirements` - Directory containing `.txt` files for different optional requirements.
* `spockflow/components` - Contains all components for the Hamilton DAG, including:
  * `scorecard` - For scorecards.
  * `tree` - For decision trees.
  * `dtable` - For decision tables.
* `spockflow/inference` - Files needed to serve the module as a live endpoint.
* `core.py` - Contains code to inject components into the Hamilton DAG.
* `exceptions.py` - Base for exceptions produced by various components.
* `nodes.py` - Core module for all components.
* `tests` - Contains unit tests for the project.

<h3 id="naming-conventions">🏷 Naming Conventions</h3>

- ✅ **Do** follow PEP 8 for naming conventions.
- ✅ **Do** use descriptive names for files and modules.
- ✅ **Do** name Python classes in `CamelCase` and functions in `snake_case`.

<h3 id="code-quality">🔍 Code Quality</h3>

- ✅ **Do** adhere to PEP 8 style guidelines.
- ✅ **Do** use `black` for automatic code formatting.

<h3 id="testing">🧪 Testing</h3>

- ✅ **Do** write tests using `pytest`.
- ✅ **Do** ensure all tests pass before submitting a Pull Request.

## 5. Pull Requests

- ✅ **Do** ensure your branch is up to date with the `develop` branch.
- ✅ **Do** ensure there are no conflicts with the `develop` branch.
- ✅ **Do** make sure all tests pass and code is formatted using `black`.
- ✅ **Do** provide a clear description of the changes and the purpose of the Pull Request.

> **TIP:** Make sure to review the existing codebase and follow the conventions used throughout the project.

---

Thank you for contributing! We appreciate your efforts to improve the project.
