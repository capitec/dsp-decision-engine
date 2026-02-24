import typing as t
import importlib
from pydantic import Field
from ped.modules.core import BaseModule, PEDNode


class HamiltonModule(BaseModule):
    """A module that builds its PEDNodes from one or more Hamilton Python modules.

    Provide Python module paths in dotted notation (e.g. 'my_package.my_module').
    On expand_nodes(), the modules are imported, a Hamilton FunctionGraph is built,
    and every computed (non-external) node is converted to a PEDNode.

    Internal dependencies (one Hamilton function depending on another) are wired
    automatically via the standard module_namespaced_nodes mechanism.  External
    inputs (function parameters with no corresponding function in the provided
    modules) remain as unresolved external dependencies and will be injected at
    execution time.

    Any namespace already applied by Hamilton decorators (e.g. @namespace) is
    preserved in the resulting PEDNode.namespace so that the fully-qualified node
    name remains `<module_name>.<hamilton_namespace>.<node_name>`.
    """

    type: t.Literal["hamilton"] = "hamilton"

    module_paths: t.List[str] = Field(
        description=(
            "Ordered list of dotted Python module paths to load into the Hamilton graph. "
            "Later modules can override nodes from earlier ones when "
            "allow_module_overrides=True. "
            "Example: ['my_package.transforms', 'my_package.features']"
        )
    )
    hamilton_config: t.Dict[str, t.Any] = Field(
        default_factory=dict,
        description=(
            "Config values forwarded to the Hamilton FunctionGraph. "
            "Used by @config.when, @config.when_not, etc."
        ),
    )
    allow_module_overrides: bool = Field(
        default=False,
        description=(
            "Allow later modules in module_paths to override nodes defined by "
            "earlier modules.  Mirrors the Hamilton FunctionGraph flag of the same name."
        ),
    )

    def expand_nodes(self) -> t.List[PEDNode]:
        from hamilton.graph import FunctionGraph
        from hamilton.node import NodeType  # noqa: F401 – used via h_node.user_defined

        # ── 1. Import all specified Python modules ────────────────────────────
        loaded_modules = [importlib.import_module(path) for path in self.module_paths]

        # ── 2. Build a Hamilton FunctionGraph ─────────────────────────────────
        fn_graph = FunctionGraph.from_modules(
            *loaded_modules,
            config=self.hamilton_config,
            allow_module_overrides=self.allow_module_overrides,
        )

        # ── 3. Convert Hamilton nodes → PEDNodes ──────────────────────────────
        ped_nodes: t.List[PEDNode] = []

        for h_node in fn_graph.nodes.values():
            # Skip external / user-input nodes.
            # These are NOT computed — they represent runtime inputs whose values
            # are injected via the graph execution inputs dict.
            # node.user_defined == True  ⟺  node_source == NodeType.EXTERNAL
            if h_node.user_defined:
                continue

            # Build input_map: each function parameter maps to itself.
            # module_namespaced_nodes will then:
            #   • remap params that resolve to another node in this module
            #     → "<module_name>.<dependency_name>"
            #   • apply any user-supplied input_mapping overrides
            #   • leave everything else as an external dependency
            input_map = {param: param for param in h_node.input_types.keys()}

            ped_nodes.append(
                PEDNode(
                    name=h_node.name,
                    callable=h_node.callable,
                    original_callable=h_node.callable,
                    # Preserve any Hamilton decorator namespace (e.g. @namespace)
                    # so module_namespaced_nodes prepends the module name on top.
                    namespace=h_node.namespace,
                    input_map=input_map,
                    extra=dict(h_node.tags) if h_node.tags else {},
                )
            )

        return ped_nodes
