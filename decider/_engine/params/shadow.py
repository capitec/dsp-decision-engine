from __future__ import annotations

import warnings
from typing import Iterable, Mapping

from decider._engine.params.models import NodeParams
from decider._engine.params.validate import _local_view
from decider.exceptions import ShadowedTableWarning


def warn_shadowed_tables(nodes: Iterable[NodeParams], doc: Mapping | None) -> None:
    """Warn once per seeded table that params document `doc` sets, naming the table, its config document and the key."""
    if not isinstance(doc, Mapping):
        return  # validation reports a document that is not a mapping
    shared = doc.get("shared")
    shared = shared if isinstance(shared, Mapping) else {}
    for node in nodes:
        local = _local_view(doc, node.path)
        for d in node.decls:
            if d.seeded_from is None or d.allow_override:
                continue
            key = "".join(f'["{part}"]' for part in (["shared", d.shared_key] if d.shared_key else [*node.path.split("/"), d.name]))
            if (d.shared_key in shared) if d.shared_key else (isinstance(local, Mapping) and d.name in local):
                warnings.warn(
                    f"params key {key} sets table `{d.name}` of {node.path}, whose default rows come from config document "
                    f"`{d.seeded_from}`; the params rows replace the document's. Remove the key to use the document, or "
                    "silence this: warnings.filterwarnings(\"ignore\", category=ShadowedTableWarning), "
                    "DECIDER_BUILD__WARN_SHADOWED_TABLES=false, or param_table(..., allow_override=True).",
                    ShadowedTableWarning)
