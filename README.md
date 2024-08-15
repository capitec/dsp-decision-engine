# DSP Decision Engine

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SpockFlow** is a Python framework designed to create standalone micro-services that enrich data with actionable outputs. It supports both batch and live inference modes, and extends existing frameworks to simplify data flows, including policy rules and scoring. Leveraging Hamilton for traceability, SpockFlow provides a powerful, modular approach for data enrichment and model deployment.

## Table of Contents

- [Introduction](docs/intro.md)
- [Installation](docs/getting_started/install.md)
- [Concepts](docs/concepts/index.md)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

SpockFlow is built to be extensible and modular, allowing the reuse of pipelines and configurations across multiple data flows. Its emphasis on runtime traceability and explainability is empowered by Hamilton, which helps track and visualize data lineage and identify process steps leading to specific outcomes.

![Example Pipeline](./docs/_static/getting-started/example_pipeline.drawio.svg)

For a more detailed introduction, see [Introduction](docs/main.rst).

## Installation

To get started with SpockFlow, you need to install the required dependencies. Follow the instructions in the [Installation Guide](docs/getting_started/install.md) to set up your environment.

```bash
pip install spockflow[all]
```

## Concepts

Explore the foundational principles and components of SpockFlow in the [Concepts](docs/concepts/index.md) section. This guide covers:

- **Decision Trees**: Automate decision-making processes based on defined conditions.
- **Decision Tables**: Map input values to outputs based on conditions.
- **Score Cards**: Assign scores to entities based on parameters.
- **API Customization**: Customize and extend SpockFlow functionalities.

## Usage Examples

Here are some examples of how to use SpockFlow:

### Decision Trees

Create and use decision trees in SpockFlow:

```python
from spockflow.components.tree import Tree, Action
from spockflow.core import initialize_spock_module
import pandas as pd
from typing_extensions import TypedDict

class Reject(TypedDict):
    code: int
    description: str

RejectAction = Action[Reject]

# Initialize Tree
tree = Tree()

# Define conditions and actions
@tree.condition(output=RejectAction(code=102, description="My first condition"))
def first_condition(d: pd.Series, e: pd.Series, f: pd.Series) -> pd.Series:
    return (d > 5) & (e > 5) & (f > 5)

tree.visualize(get_value_name=lambda x: x["description"][0])
```

For more details and advanced usage, check out the [Concepts](docs/concepts/index.md) section.

## Contributing

We welcome contributions to SpockFlow! Please refer to our [Contributing Guide](CONTRIBUTING.md) for information on how to contribute.

- **Fork the repository** and create a branch from `develop`.
- **Install dependencies** using `pip install -r requirements/all.txt`.
- **Run tests** with `pytest` to ensure everything is working.
- **Submit a Pull Request** with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for your interest in SpockFlow! We look forward to your contributions and feedback.
