"""
Walk a BaseModule tree and produce graph structures for the visualiser.

Two graph types:
  build_graph(module)            — module-level structural graph (pipeline view)
  build_expression_graph(module) — expression node DAG inside one ExpressionModule

Both return ModuleGraph.  The module_ref on each GraphNode holds the live
module object so the app can drill into it.
"""

import typing as t
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    label: str
    kind: str          # "expression" | "sequential" | "join" | "union" | "col" | "config" | "unknown"
    type_id: str
    parent: t.Optional[str] = None
    fields: t.Dict[str, t.Any] = field(default_factory=dict)
    module_ref: t.Any = None       # live BaseModule if drillable
    drillable: bool = False


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""


@dataclass
class ModuleGraph:
    nodes: t.List[GraphNode] = field(default_factory=list)
    edges: t.List[GraphEdge] = field(default_factory=list)

    def to_graphviz(self) -> "graphviz.Digraph":
        import graphviz
        dot = graphviz.Digraph(graph_attr={"rankdir": "TB", "splines": "ortho"})

        _KIND_COLOURS = {
            "expression": "#4C9BE8",
            "sequential": "#E8884C",
            "join":       "#4CE8A0",
            "union":      "#9B4CE8",
            "col":        "#888888",
            "config":     "#C8A850",
            "unknown":    "#AAAAAA",
        }

        for n in self.nodes:
            colour = _KIND_COLOURS.get(n.kind, "#AAAAAA")
            tooltip = "\n".join(f"{k}: {v}" for k, v in n.fields.items()) or n.type_id
            shape = "ellipse" if n.kind in ("col", "config") else "box"
            border = "bold" if n.drillable else ""
            dot.node(
                n.id,
                label=n.label,
                shape=shape,
                style=f"filled,rounded,{border}".strip(","),
                fillcolor=colour,
                fontcolor="white",
                tooltip=tooltip,
            )
        for e in self.edges:
            dot.edge(e.source, e.target, label=e.label)
        return dot


# ── helpers ───────────────────────────────────────────────────────────────────

def _kind(module) -> str:
    type_id = getattr(module, "type", "")
    if type_id == "sequential":
        return "sequential"
    if type_id == "join":
        return "join"
    if type_id == "union":
        return "union"
    if hasattr(module, "expand_nodes"):
        return "expression"
    return "unknown"


def _config_fields(module) -> t.Dict[str, t.Any]:
    _SKIP = {"type", "name", "steps", "modules", "left", "right", "on", "how"}
    try:
        raw = module.model_dump(exclude_defaults=False)
    except Exception:
        return {}
    return {k: v for k, v in raw.items() if k not in _SKIP and not k.startswith("_")}


# ── module-level graph ────────────────────────────────────────────────────────

def _walk(
    module,
    graph: ModuleGraph,
    parent_id: t.Optional[str] = None,
    counter: t.Optional[t.List[int]] = None,
) -> str:
    if counter is None:
        counter = [0]

    counter[0] += 1
    node_id = f"node_{counter[0]}"
    type_id = getattr(module, "type", type(module).__name__)
    name = getattr(module, "name", type_id)
    kind = _kind(module)
    drillable = kind in ("expression", "sequential", "join", "union")

    graph.nodes.append(GraphNode(
        id=node_id,
        label=name,
        kind=kind,
        type_id=type_id,
        parent=parent_id,
        fields=_config_fields(module),
        module_ref=module,
        drillable=drillable,
    ))

    if parent_id is not None:
        graph.edges.append(GraphEdge(source=parent_id, target=node_id))

    if kind == "sequential":
        prev = node_id
        for step in module.steps:
            child_id = _walk(step, graph, parent_id=node_id, counter=counter)
            if graph.edges and graph.edges[-1].source == node_id:
                graph.edges[-1] = GraphEdge(source=prev, target=child_id, label="then")
            prev = child_id

    elif kind == "join":
        for side, ref in (("left", module.left), ("right", module.right)):
            if hasattr(ref, "type"):
                child_id = _walk(ref, graph, parent_id=node_id, counter=counter)
                if graph.edges:
                    graph.edges[-1].label = side
            else:
                fid = f"frame_{ref}_{counter[0]}"
                counter[0] += 1
                graph.nodes.append(GraphNode(
                    id=fid, label=f'"{ref}"', kind="col", type_id="frame", parent=node_id,
                ))
                graph.edges.append(GraphEdge(source=node_id, target=fid, label=side))

    elif kind == "union":
        for child_mod in module.modules:
            _walk(child_mod, graph, parent_id=node_id, counter=counter)

    return node_id


def build_graph(module) -> ModuleGraph:
    """Module-level structural graph for any BaseModule tree."""
    g = ModuleGraph()
    _walk(module, g)
    return g


# ── expression node DAG ───────────────────────────────────────────────────────

def build_expression_graph(module) -> ModuleGraph:
    """
    Return a computation DAG for an ExpressionModule showing individual
    function nodes, their column inputs, and config injections.
    """
    from decider.modules.expression import ExternalInputNode, StaticValueNode, Node as ExprNode

    g = ModuleGraph()
    nodes = module.expand_nodes()

    # add a function node for every expression node
    for name, expr_node in nodes.items():
        g.nodes.append(GraphNode(
            id=f"fn_{name}",
            label=name,
            kind="expression",
            type_id="function",
            drillable=False,
        ))

    # add edges: inputs → function nodes
    for name, expr_node in nodes.items():
        for param, ref in expr_node.input_map.items():
            if isinstance(ref, ExprNode):
                g.edges.append(GraphEdge(source=f"fn_{ref.name}", target=f"fn_{name}", label=param))
            elif isinstance(ref, ExternalInputNode):
                col_id = f"col_{ref.input_name}"
                if not any(n.id == col_id for n in g.nodes):
                    g.nodes.append(GraphNode(
                        id=col_id,
                        label=ref.input_name,
                        kind="col",
                        type_id="column",
                        drillable=False,
                    ))
                g.edges.append(GraphEdge(source=col_id, target=f"fn_{name}", label=param))
            elif isinstance(ref, StaticValueNode):
                val = ref.value
                cfg_id = f"cfg_{name}_{param}"
                # show the config type name, not the full repr
                cfg_label = type(val).__name__ if hasattr(val, "__class__") else str(val)
                if not any(n.id == cfg_id for n in g.nodes):
                    g.nodes.append(GraphNode(
                        id=cfg_id,
                        label=cfg_label,
                        kind="config",
                        type_id="config",
                        drillable=False,
                        fields=val.model_dump() if hasattr(val, "model_dump") else {},
                    ))
                g.edges.append(GraphEdge(source=cfg_id, target=f"fn_{name}", label=param))

    return g


# ── intermediate value extraction ─────────────────────────────────────────────

def run_with_intermediates(
    module,
    inputs: t.Dict[str, "pl.DataFrame"],
) -> t.List[t.Tuple[str, "pl.DataFrame"]]:
    """
    Execute module and return a list of (label, DataFrame) pairs, one per
    logical step, in execution order.

    - ExpressionModule  → one entry per compiled expression column, accumulated
    - SequentialModule  → one entry per step
    - Others            → single entry with final output
    """
    import polars as pl

    kind = _kind(module)

    if kind == "expression":
        return _run_expression_intermediates(module, inputs)
    elif kind == "sequential":
        return _run_sequential_intermediates(module, inputs)
    elif kind == "join":
        return _run_join_intermediates(module, inputs)
    else:
        out = module(inputs)
        if isinstance(out, pl.LazyFrame):
            out = out.collect()
        return [(getattr(module, "name", "output"), out)]


def _run_expression_intermediates(
    module,
    inputs: t.Dict[str, "pl.DataFrame"],
) -> t.List[t.Tuple[str, "pl.DataFrame"]]:
    import polars as pl

    module.compile_expressions()
    ce = module._compiled_expressions
    frame = inputs.get(ce.input_frame)
    if frame is None:
        frame = next(iter(inputs.values()))
    if isinstance(frame, pl.DataFrame):
        frame = frame.lazy()

    results = []
    accumulated = frame
    for col_name, expr in ce.expressions.items():
        accumulated = accumulated.with_columns(expr.alias(col_name))
        snapshot = accumulated.collect()
        results.append((col_name, snapshot))

    return results


def _run_sequential_intermediates(
    module,
    inputs: t.Dict[str, "pl.DataFrame"],
) -> t.List[t.Tuple[str, "pl.DataFrame"]]:
    import polars as pl

    frames = {
        k: v.lazy() if isinstance(v, pl.DataFrame) else v
        for k, v in inputs.items()
    }
    current = frames.get("input") if "input" in frames else next(iter(frames.values()))

    results = []
    for step in module.steps:
        frames["input"] = current
        out = step(frames)
        if isinstance(out, pl.LazyFrame):
            out = out.collect()
        current = out.lazy()
        results.append((getattr(step, "name", step.type), out))

    return results


def _run_join_intermediates(
    module,
    inputs: t.Dict[str, "pl.DataFrame"],
) -> t.List[t.Tuple[str, "pl.DataFrame"]]:
    import polars as pl

    out = module(inputs)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    return [(getattr(module, "name", "join"), out)]
