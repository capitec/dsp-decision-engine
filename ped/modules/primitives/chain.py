import typing as t
from dataclasses import replace
from pydantic import field_validator, model_validator
from ped.modules.core import BaseModule, PEDNode


class ChainModule(BaseModule):
    type: t.Literal["chain"]
    modules: t.List[BaseModule]

    @field_validator("modules", mode="after")
    @classmethod
    def validate_unique_names(cls, modules: t.List[BaseModule]) -> t.List[BaseModule]:
        names = [m.name for m in modules]
        seen: set = set()
        duplicates = {n for n in names if n in seen or seen.add(n)}  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(
                f"Module names within a chain must be unique. Duplicates found: {duplicates}"
            )
        return modules

    @model_validator(mode="after")
    def validate_chain_connections(self) -> "ChainModule":
        for i in range(len(self.modules) - 1):
            current = self.modules[i]
            nxt = self.modules[i + 1]
            if current.output_name is None:
                raise ValueError(
                    f"Module '{current.name}' (position {i}) cannot be chained: "
                    f"it has no output_name defined."
                )
            if nxt.input_name is None:
                raise ValueError(
                    f"Module '{nxt.name}' (position {i + 1}) cannot be chained: "
                    f"it has no input_name defined."
                )
        return self

    @classmethod
    def from_modules(cls, modules: t.List[BaseModule], name: t.Optional[str] = None) -> "ChainModule":
        return cls(name=name or modules[0].name, modules=modules)

    def with_name(self, name: str) -> "ChainModule":
        """Return a copy of this chain with a new name."""
        return self.model_copy(update={"name": name})

    @staticmethod
    def _get_previous_output_node(module_nodes: t.List[PEDNode], output_name: t.Optional[str]) -> t.Optional[PEDNode]:
        if output_name is None: return None
        output_nodes = [v for v in module_nodes if v.name == output_name]
        # TODO make this error better
        assert len(output_nodes) <= 1, f"Module should have only one node with the name matching the modules output name"
        return output_nodes[0] if len(output_nodes) else None


    def expand_nodes(self) -> t.List[PEDNode]:
        """
        Expand all modules into namespaced PEDNodes, wiring each module's output_name
        to the next module's input_name via input_mapping.
        """
        all_nodes: t.List[PEDNode] = []
        if len(self.modules) == 0:
            return all_nodes
        
        prev_module = self.modules[0]
        module_nodes = prev_module.module_namespaced_nodes()
        all_nodes.extend(module_nodes)

        for i, module in enumerate(self.modules[1:], start=1):
            prev_output_node = self._get_previous_output_node(
                # This is still the previous module_nodes
                # Dont think its needed to keep track of the prev nodes after mapping inputs as it doesnt change output names
                # However if there is reason that can be changed later
                module_nodes, 
                prev_module.output_name
            )
            prev_output_name = prev_output_node.namespaced_name
            # NOTE: the input_name shouldnt be namespaced because the node should always view it as an external input
            current_module_input = module.input_name
            # Defer this check till now because if its the last in the chain it doesnt need an output name
            if not prev_output_name or not current_module_input:
                raise ValueError(
                    f"Modules in a chain must have output_name and input_name defined. "
                    f"Module '{self.modules[i-1].name}' has output_name='{prev_module_output}'. "
                    f"Module '{module.name}' has input_name='{current_module_input}'."
                )
            
            module_nodes = module.module_namespaced_nodes()
            # Patch the input mapping of the current module's nodes to connect to the previous module's output
            mapped_node = False
            for node in module_nodes:

                new_input_map = {}
                # This probably couldve been done with dict comprehension
                # But i wanted to tack mapped_node for safety
                for k,v in node.input_map.items():
                    if v == current_module_input:
                        mapped_node = True
                        new_input_map[k] = prev_output_name
                    else:
                        new_input_map[k] = v
                node = replace(node, input_map=new_input_map)
                all_nodes.append(node)

            if not mapped_node:
                raise ValueError(
                    f"Module '{module.name}' does not have an input that matches the module's input name. "
                    f"Expected input name: '{current_module_input}'."
                )

        return all_nodes
