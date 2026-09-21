import typing as t
from collections import deque


class HasDependencies(t.Protocol):
    name: str

    def get_dependencies(self) -> t.List[str]: ...


T = t.TypeVar("T", bound=HasDependencies)


def topological_sort(nodes: t.List[T]) -> t.List[T]:
    """Kahn's algorithm. Works on any node type that implements HasDependencies."""
    node_map = {n.name: n for n in nodes}
    in_degree = {n.name: 0 for n in nodes}
    adjacency: t.Dict[str, t.List[str]] = {n.name: [] for n in nodes}

    for node in nodes:
        for dep in node.get_dependencies():
            if dep in node_map:
                adjacency[dep].append(node.name)
                in_degree[node.name] += 1

    queue = deque(name for name, deg in in_degree.items() if deg == 0)
    result = []
    while queue:
        current = queue.popleft()
        result.append(current)
        for neighbour in adjacency[current]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(result) != len(nodes):
        cycle = [n for n, d in in_degree.items() if d > 0]
        raise ValueError(f"Circular dependency detected: {cycle}")

    return [node_map[name] for name in result]
