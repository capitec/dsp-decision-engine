from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, TypeAdapter

from decikit._settings import get_settings
from decikit.lab.checks._base import STANDARD, Check
from decikit.lab.checks._report import Severity
from decikit.lab.data._providers import project_root
from decikit.exceptions import DecikitError, wrap_import_errors
from decikit._registry import ref

CheckRef = ref(Check)
_CHECK = TypeAdapter(CheckRef)


def build(item: Any) -> list[Check]:
    """Check instances for one entry of a check list: an instance, an alias, a config dict or a nested set."""
    if isinstance(item, str):
        return [Check.resolve(item)()]
    if isinstance(item, (list, tuple, set, frozenset)):
        return [c for each in item for c in build(each)]
    return [_CHECK.validate_python(item)]


class ChecksConfig(BaseModel):
    """The project's check policy (`checks.yaml`): what runs, how severe, what is off, and the gate."""

    model_config = ConfigDict(extra="forbid")

    extends: Literal["standard"] | None = "standard"
    checks: list[CheckRef] = []
    disable: list[str] = []
    fail_on: Severity = Severity.ERROR

    def instances(self) -> list[Check]:
        # An entry replaces the same type from `extends`, so `{type: durable_ids, severity: error}` promotes it.
        # Entries of one type with different options all run; only identical ones are one.
        own = {(c.type, c.model_dump_json()): c for c in self.checks}
        base = [c for c in build(STANDARD if self.extends else ()) if c.type not in {t for t, _ in own}]
        return [c for c in (*base, *own.values()) if c.type not in self.disable and c.code not in self.disable]


def find_config(start: str | Path | None = None) -> Path | None:
    """The project's checks file (`checks_path`, or its `.json` / `.yml` sibling), or `None`."""
    text = get_settings().checks_path
    if "${PROJECT_ROOT}" in text:
        root = project_root(start or Path.cwd()) or Path.cwd()
        text = text.replace("${PROJECT_ROOT}", str(root))
    path = Path(text).resolve()
    return next((p for p in (path, path.with_suffix(".json"), path.with_suffix(".yml")) if p.is_file()), None)


def load_config(path: str | Path) -> ChecksConfig:
    """Read and validate a `checks.yaml` / `checks.json`; every problem is a `DecikitError` naming the file."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise DecikitError(f"{path}: cannot read checks config: {e}") from e
    if path.suffix == ".json":
        doc = json.loads(text)
    else:
        with wrap_import_errors("yaml"):
            import yaml
        doc = yaml.safe_load(text)
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise DecikitError(f"{path}: expected a mapping with extends/checks/disable/fail_on, got {type(doc).__name__}")
    try:
        config = ChecksConfig.model_validate(doc)
    except ValidationError as e:
        raise DecikitError(f"{path}: invalid checks config:\n{e}") from e
    except ValueError as e:
        raise DecikitError(f"{path}: invalid checks config: {e}") from e
    return config
