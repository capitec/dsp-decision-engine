import typing as t
from logging import getLogger
from .edges import Edge


logger = getLogger(__name__)


def _build_child_mapping(edges: t.List[Edge]) -> t.Dict[str, t.Dict[int, str]]:
    """Build child node mapping from edges array."""
    child_map = {}
    for edge in edges:
        source = edge.source
        target = edge.target
        source_indexes = edge.data.sourceIndex
        if not isinstance(source_indexes, list):
            source_indexes = [source_indexes]
        for source_index in source_indexes:
            if source not in child_map:
                child_map[source] = {}
            if source_index in child_map[source]:
                if child_map[source][source_index] == target:
                    logger.warning(
                        f"Duplicate edge found {source}:{source_index} -> {target}"
                    )
                else:
                    raise ValueError(
                        f"Conflicting edge found {source}:{source_index} -> {target} != {child_map[source][source_index]}"
                    )
            child_map[source][source_index] = target
    return child_map
