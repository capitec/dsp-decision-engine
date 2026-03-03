import typing as t
import importlib
from pydantic import Field, PrivateAttr, model_validator
from ped.modules.core import BaseModule, PEDNode
from ._importer import add_to_path

if t.TYPE_CHECKING:
    from types import ModuleType
    from hamilton.node import NodeType


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

    type: t.Literal["hamilton"]

    module_paths: t.List[str] = Field(
        description=(
            "Ordered list of dotted Python module paths to load into the Hamilton graph. "
            "Later modules can override nodes from earlier ones when "
            "allow_module_overrides=True. "
            "Example: ['my_package.transforms', 'my_package.features']"
        )
    )
    base_import_path: t.Optional[str] = Field(
        default=None,
        description=(
            "The location to load into sys.paths before importing the modules"
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

    _hamilton_nodes: t.List["NodeType"] = PrivateAttr()  # for caching imported modules

    @staticmethod
    def _import_modules(base_import_path: str, module_paths: t.List[str]) -> t.List["ModuleType"]:
        """Import the specified modules and return the module objects."""
        with add_to_path(base_import_path):
            return [importlib.import_module(path) for path in module_paths]
    
    @staticmethod
    def _build_hamilton_graph(modules: t.List["ModuleType"]) -> t.List["NodeType"]:
        """Build the Hamilton FunctionGraph from the imported modules."""
        from hamilton.driver import Builder
        graph = Builder().with_modules(*modules).build().graph
        return list(graph.nodes.values())

    @model_validator(mode="after")
    def validate_can_build_graph(self) -> "t.Self":
        """Validate that we can successfully import the modules and build a Hamilton graph."""
        try:
            modules = self._import_modules(self.base_import_path, self.module_paths)
        except ModuleNotFoundError as e:
            raise ValueError(f"Failed to import {e.name} using additional path={self.base_import_path}. Please ensure you have configured the modules and the base import path correctly. Detail:\n{e}") from e
        except ImportError as e:
            raise ValueError(f"Failed to import modules. Detail:\n{e}") from e

        try:
            self._hamilton_nodes = self._build_hamilton_graph(modules)
        except Exception as e:
            raise ValueError(f"Failed to build Hamilton graph. Detail:\n{e}") from e
        
        return self

    def expand_nodes(self) -> t.List[PEDNode]:
        """Convert the Hamilton graph nodes to PEDNodes, preserving namespaces and wiring internal dependencies."""
        ped_nodes: t.List[PEDNode] = []
        for h_node in self._hamilton_nodes:
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
