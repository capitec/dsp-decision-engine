import typing as t
from pydantic import Field, field_validator
from ped.modules.core import BaseModule, PEDNode
from ped.modules._ext import GraphModule

if t.TYPE_CHECKING:
    from .external import ExternalModule

class NamespacedModule(BaseModule):
    """Groups a set of child modules under a single namespace.

    All child nodes are placed under ``<module_name>.<child_name>.<node_name>``.

    Nodes whose names appear in ``outputs`` are *promoted*: they are lifted to
    ``<module_name>.<node_name>`` (one level up, still inside this module's
    namespace but no longer inside any child-module sub-namespace).  Any other
    node that depends on a promoted node has its reference updated automatically.

    Example
    -------
    Given::

        Namespaced(
            name="scoring",
            modules=[ScoreCard(name="sc", ...), ProbabilityDefault(name="pd", ...)],
            outputs=["output"],        # pd's output node
        )

    Resulting fully-qualified node names (assuming the adaptor calls
    ``module_namespaced_nodes("scoring")``):

    * ``scoring.sc.output``  – ScoreCard's output (not promoted)
    * ``scoring.output``     – ProbabilityDefault's output  (promoted)
    """

    type: t.Literal["namespaced"]

    modules: t.List[GraphModule]
    outputs: t.List[str] = Field(
        default_factory=list,
        description=(
            "Node names to promote to the top of this module's namespace. "
            "Specify as the bare node name (e.g. 'output') if unambiguous, "
            "or as the fully-qualified child path (e.g. 'pd.output') to "
            "disambiguate when multiple child modules expose the same name."
        ),
    )

    @field_validator("modules", mode="after")
    @classmethod
    def validate_unique_names(cls, modules: t.List[GraphModule]) -> t.List[GraphModule]:
        names = [m.root.name for m in modules]
        seen: set = set()
        duplicates = {n for n in names if n in seen or seen.add(n)}  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(
                f"Module names within a Namespaced block must be unique. "
                f"Duplicates found: {duplicates}"
            )
        return modules

    # ------------------------------------------------------------------ #
    # Abstract method implementation – used internally by                 #
    # module_namespaced_nodes below.                                       #
    # ------------------------------------------------------------------ #

    def expand_nodes(self) -> t.List[PEDNode]:
        """Expand all child modules with their own sub-namespaces applied.

        Returns nodes whose ``namespace`` already contains the child module
        name, e.g. ``("child_name",)``.  This is consumed by the overridden
        ``module_namespaced_nodes`` and should not be called externally.
        """
        all_nodes: t.List[PEDNode] = []
        for module in self.modules:
            all_nodes.extend(module.root.module_namespaced_nodes())
        return all_nodes
    
    def as_external_module(
        self,
        module_name: str = None,
        config_path: str = None,
    ) -> "ExternalModule":
        """Convert this NamespacedModule to an ExternalModule by serializing its config
        and storing it in a temporary JSON file, then returning an ExternalReference
        to that file. This is used by the Hamilton adaptor to convert nested modules
        into a flat list of ExternalReferences.
        """
        from .external import ExternalModule, ExternalReference
        
        # Serialize this module's config to a dict
        config_dict = self.model_dump(exclude={"type"})


        ref = ExternalReference(
            module_name=module_name or self.name,
            **({} if config_path is None else {"config_path": config_path})
        )
        
        # Create an ExternalModule with this config as its content
        external_module = ExternalModule(
            **config_dict,
            ref=ref,
        )
        
        return external_module

    # # ------------------------------------------------------------------ #
    # # Override: two-level namespacing + output promotion                  #
    # # ------------------------------------------------------------------ #

    # def module_namespaced_nodes(self, module_name: str) -> t.List[PEDNode]:
    #     """Namespace all child nodes under ``module_name`` and promote outputs.

    #     We override the base implementation because ``expand_nodes()`` already
    #     returns nodes with child-level namespaces applied (refs look like
    #     ``"child.dep"``), which the base ``node_map`` lookup (keyed by bare
    #     name) cannot resolve correctly.
    #     """
    #     # ── 1. Expand children with their own sub-namespaces ──────────────
    #     child_nodes = self.expand_nodes()

    #     # Map  child-level namespaced_name  →  node  for internal-dep detection
    #     # e.g. "sc.output" → <PEDNode name="output" namespace=("sc",)>
    #     child_ns_map: t.Dict[str, PEDNode] = {
    #         node.namespaced_name: node for node in child_nodes
    #     }

    #     # ── 2. Resolve which nodes are promoted outputs ────────────────────
    #     promoted_child_ns_names: t.Set[str] = set()
    #     for spec in self.outputs:
    #         matching = [
    #             n for n in child_nodes
    #             if n.name == spec or n.namespaced_name == spec
    #         ]
    #         if not matching:
    #             available = [n.namespaced_name for n in child_nodes]
    #             raise ValueError(
    #                 f"Output '{spec}' not found in Namespaced module '{self.name}'. "
    #                 f"Available node paths: {available}"
    #             )
    #         if len(matching) > 1:
    #             found = [n.namespaced_name for n in matching]
    #             raise ValueError(
    #                 f"Ambiguous output '{spec}' in Namespaced module '{self.name}': "
    #                 f"{found}. Use the fully-qualified child path to disambiguate."
    #             )
    #         promoted_child_ns_names.add(matching[0].namespaced_name)

    #     # Helper: given a child-level namespaced_name, return the final
    #     # fully-qualified name after the outer namespace is applied.
    #     def final_fqn(child_ns_name: str, node: PEDNode) -> str:
    #         if child_ns_name in promoted_child_ns_names:
    #             # Promoted → lives at  module_name.node_name
    #             return f"{module_name}.{node.name}"
    #         else:
    #             # Regular → lives at  module_name.child_ns.node_name
    #             return node.namespaced_name_with_namespace(module_name)

    #     # ── 3. Build final PEDNodes ────────────────────────────────────────
    #     final_nodes: t.List[PEDNode] = []
    #     for child_node in child_nodes:
    #         is_promoted = child_node.namespaced_name in promoted_child_ns_names

    #         # Namespace: promoted nodes sit one level higher
    #         new_namespace = (
    #             (module_name,)
    #             if is_promoted
    #             else (module_name,) + child_node.namespace
    #         )

    #         # Rewrite input_map references
    #         updated_input_map: t.Dict[str, str] = {}
    #         for param, ref in child_node.input_map.items():
    #             if ref in child_ns_map:
    #                 # Internal dep → compute its final FQN (accounts for promotion)
    #                 updated_input_map[param] = final_fqn(ref, child_ns_map[ref])
    #             else:
    #                 # External dep → honour self.input_mapping overrides
    #                 updated_input_map[param] = self.input_mapping.get(ref, ref)

    #         final_nodes.append(
    #             PEDNode(
    #                 name=child_node.name,
    #                 callable=child_node.callable,
    #                 original_callable=child_node.original_callable,
    #                 namespace=new_namespace,
    #                 input_map=updated_input_map,
    #                 static_kwargs=child_node.static_kwargs,
    #                 extra=child_node.extra,
    #             )
    #         )

    #     return final_nodes
