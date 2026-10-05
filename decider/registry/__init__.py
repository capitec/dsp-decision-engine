"""Tag-addressed registries for pluggable types.

An installed package can ship a type without the user importing it: declare
the class under the root's entry-point group, named by its `type` tag.
`ConfigurableStep.load({"type": "scaled", ...})` then imports the package on
the first miss. Groups: `decider.steps`, `decider.config_stores`,
`decider.metrics`, `decider.plots`, `decider.blocks`, `decider.datasets`
(and `decider.checks`).

Example (the package's `pyproject.toml`)::

    [project.entry-points."decider.steps"]
    scaled = "credit_risk.steps:Scaled"
"""
from __future__ import annotations

from ._core import BaseRegistryModule, import_path, ref

__all__ = ["BaseRegistryModule", "import_path", "ref"]
