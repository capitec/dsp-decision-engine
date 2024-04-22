import typing

from dataclasses import dataclass, field

if typing.TYPE_CHECKING:
    from hamilton import node

InputTypes = typing.Dict[str, typing.Union[typing.Type, typing.Tuple[typing.Type, "node.DependencyType"]]]


# Dataclass here allows simple variable node to be a dataclass
@dataclass
class VariableNodeExpander():
    def __post_init__(self):
        self._name = None
        self._module = None

    @property
    def name(self):
        if self._name is None: raise ValueError("Node must be generated before a name can be called")
        return self._name
    
    def set_name(self, name):
        if name is None: return
        if self._name is not None and name != self._name:
            raise ValueError(f"Variable node being accessed with multiple names. Please use {name} = calculate(lambda: {self._name}) to make a copy with a different name.")
        self._name = name

    @property
    def module(self):
        if self._module is None: raise ValueError("Node must be generated before a module can be called")
        return self._module
    
    def set_module(self, module):
        if module is None: return
        if self._module is not None and module != self._module:
            raise ValueError(f"Variable node being accessed with multiple modules. Please use {module} = calculate(lambda: {self._module}) to make a copy with a different module.")
        self._module = module

    def get_return_type(self): ...
    def generate_nodes(self, config: dict, var_name: str=None) -> typing.List["node.Node"]: ...



@dataclass
class SimpleVariableNode(VariableNodeExpander):
    function: typing.Callable
    return_type: typing.Any
    doc: typing.Optional[str] = None
    module: typing.Optional[str] = None
    additional_tags: typing.Dict[str, typing.Any] = field(default_factory=dict)
    input_types: InputTypes = None

    def __post_init__(self):
        super().__post_init__()
        self._rule_engine_internal_prop_name_ = f"VariableNodeInstance {id(self)}"

    def get_return_type(self):
        return self.return_type

    def generate_nodes(self, config: dict, var_name: str=None) -> typing.List["node.Node"]:
        import inspect
        from hamilton import node

        self.set_name(var_name)
        fn_module = inspect.getmodule(self.function).__name__
        return [node.Node(
                self.name,
                self.return_type,
                self.doc,
                callabl=self.function,
                tags={
                    "module": self.module or fn_module, 
                    **self.additional_tags
                },
                node_source=node.NodeType.STANDARD,
                input_types = self.input_types
            )]
    
    def get_str(self):
        from tracing import TracerSeries
        if self.doc is not None:
            return self.doc
        subbed_values = {}
        for k, v in self.input_types.items():
            # TODO maybe different tracers for different types
            # if isinstance(v, pd.Series):
            #     v = TracerSeries()
            subbed_values[k] = TracerSeries(k, is_base=True)
        res = self.function(**subbed_values)
        return res._name
