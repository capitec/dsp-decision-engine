import typing as t
from pydantic import field_validator, model_validator
from ped.modules.core import BaseModule, PEDNode


class ChainModule(BaseModule):
    type: t.Literal["chain"] = "chain"
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

    def expand_nodes(self) -> t.List[PEDNode]:
        """
        Expand all modules into namespaced PEDNodes, wiring each module's output_name
        to the next module's input_name via input_mapping.
        """
        all_nodes: t.List[PEDNode] = []

        for i, module in enumerate(self.modules):
            # Build the cross-module input_mapping for this module:
            # if there is a previous module, map this module's input_name to the
            # previous module's namespaced output node.
            extra_input_mapping: t.Dict[str, str] = dict(module.input_mapping)

            if i > 0:
                prev = self.modules[i - 1]
                # prev.output_name is guaranteed non-None by the validator
                prev_namespaced_output = f"{prev.name}.{prev.output_name}"
                # current module's input_name is guaranteed non-None by the validator
                extra_input_mapping[module.input_name] = prev_namespaced_output  # type: ignore[index]

            # Temporarily patch input_mapping so module_namespaced_nodes picks it up
            patched = module.model_copy(update={"input_mapping": extra_input_mapping})
            all_nodes.extend(patched.module_namespaced_nodes(module.name))

        return all_nodes
