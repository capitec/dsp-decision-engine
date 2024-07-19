# Spockflow Installation Guide

Welcome to the installation guide for Spockflow. Spockflow is a powerful package designed to streamline decisioning processes with its cutting-edge features.

## Prerequisites

Before you begin, ensure you have the following prerequisites installed:
- Python (version 3.8 or higher recommended)
- Git (if installing from source)
- AWS CLI configured with appropriate credentials for AWS CodeArtifact (if applicable)
    - Alternatively be setup in a notebook environment on the Data Science Platform

## Installation Options

### Option 1: Pip Install

To install Spockflow via pip, use the following command:

```bash
pip install spockflow
```

#### Installing with Optional Features
**YAML Support**: To add YAML support, use the [yaml] extra option:

```bash
pip install spockflow[yaml]
```
**Web Application (REST API)**: To enable local REST API serving, use the [webapp] extra option:
```
pip install spockflow[webapp]
```

## Option 2: Install from Source
To install Spockflow from source, follow these steps:

Clone the repository from GitHub:

```bash
git clone https://github.com/spockflow/spockflow.git
cd spockflow
pip install -e .
```

## Verify Installation
To verify that Spockflow has been installed correctly, you can run the following command to check the version:

```bash
 python3 -c "import spockflow; print(spockflow.__version__)"
```

## Usage
Refer to the Spockflow documentation for detailed usage instructions and examples.

## Troubleshooting
If you encounter any issues during installation or usage, please refer to the FAQs or reach out to our support team at sholto.armstrong@capitecbank.co.za.

