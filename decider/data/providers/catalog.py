from __future__ import annotations

import json
from pathlib import Path

from decider._settings import get_settings
from decider.data.providers.base import DatasetError, DatasetProvider

_FILE_SUFFIXES = (".csv", ".parquet", ".jsonl", ".ndjson", ".json")


def project_root(start) -> Path | None:
    """The nearest ancestor of `start` (a file or a directory) holding a `configs/` directory, as `decider serve` finds one."""
    path = Path(start).resolve()
    directory = path if path.is_dir() else path.parent
    for candidate in [directory, *directory.parents]:
        if (candidate / "configs").is_dir():
            return candidate
    return None


def dataset_documents(start=None, path=None) -> dict[str, dict]:
    """The datasets folder's provider documents by name, unvalidated: inferred from its files, then `config.json`.

    Example::

        dataset_documents()   # {"loans": {"type": "file", "path": "/proj/datasets/loans.csv"}}
    """
    folder = datasets_folder(start, path)
    if not folder.is_dir():
        return {}
    docs: dict[str, dict] = {}
    for entry in sorted(folder.iterdir()):
        if entry.name.startswith((".", "_")) or entry.name == "config.json":
            continue
        if entry.is_dir():
            name, doc = entry.name, {"type": "folder", "path": str(entry)}
        elif entry.suffix.lower() in _FILE_SUFFIXES:
            name, doc = entry.stem, {"type": "file", "path": str(entry)}
        else:
            continue
        if name in docs:
            raise DatasetError(f"datasets {docs[name]['path']!r} and {doc['path']!r} are both named {name!r}")
        docs[name] = doc
    config = folder / "config.json"
    if config.is_file():
        for name, doc in json.loads(config.read_text()).items():
            if isinstance(doc, dict) and isinstance(doc.get("path"), str) and "://" not in doc["path"]:
                doc = {**doc, "path": str(folder / doc["path"])}
            docs[name] = doc
    return docs


def datasets(start=None, path=None) -> dict[str, DatasetProvider]:
    """The named datasets of the project, validated.

    `start` is a file or directory that locates the project (default: the cwd); `path`
    overrides the `datasets_path` setting.

    Example::

        datasets()["loans"].load()
    """
    found, bad = {}, []
    for name, doc in dataset_documents(start, path).items():
        try:
            found[name] = _validated(name, doc)
        except Exception as e:
            bad.append(f"{name}: {e}")
    if bad:
        raise DatasetError("invalid datasets:\n" + "\n".join(bad))
    return found


def dataset(name, start=None, path=None) -> DatasetProvider:
    """One named dataset, validated on its own so a broken sibling entry doesn't matter.

    Example::

        dataset("loans").load()
    """
    docs = dataset_documents(start, path)
    if name not in docs:
        raise DatasetError(f"unknown dataset {name!r}; known: {', '.join(docs) or 'none'}")
    try:
        return _validated(name, docs[name])
    except Exception as e:
        raise DatasetError(f"invalid dataset {name!r}: {e}") from e


def _validated(name, doc) -> DatasetProvider:
    if not isinstance(doc, dict):
        raise DatasetError(f"expected a provider document, got {doc!r}")
    provider = DatasetProvider.resolve(str(doc.get("type"))).model_validate(doc)
    provider._name = name
    return provider


def datasets_folder(start=None, path=None) -> Path:
    text = path or get_settings().datasets_path
    if "${PROJECT_ROOT}" in text:
        root = project_root(start or Path.cwd()) or Path.cwd()
        text = text.replace("${PROJECT_ROOT}", str(root))
    return Path(text).resolve()
