import typing as t
import inspect
from pydantic import BaseModel
from types import ModuleType
from .core import BaseModule, PEDNode


def generate_from_functions(module_name: str, *functions: t.Callable) -> t.Type[BaseModule]:
    """Dynamically generate a BaseModule subclass from a list of functions."""
    # Create a dictionary of the functions to be used as the namespace for the new class

    # 1. look in all functions for a keyword argument named config and ensure they are all the same
    config_class = None
    requires_injection: t.List[bool] = []
    for func in functions:
        sig = inspect.signature(func)
        if "config" not in sig.parameters:
            requires_injection.append(False)
            continue
        requires_injection.append(True)
        param = sig.parameters["config"]
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Function {func.__name__} has a 'config' parameter but it is not annotated with a type")
        if config_class is None:
            config_class = param.annotation
        elif config_class != param.annotation:
            raise TypeError(f"All functions must have the same type for the 'config' parameter. Found {config_class} and {param.annotation}")
    
    # check config is a pydantic model
    if config_class is not None:
        if not issubclass(config_class, BaseModel):
            raise TypeError(f"The 'config' parameter must be a subclass of pydantic.BaseModel. Found {config_class}")
    
    if config_class is None: config_class = BaseModel

    module_name = module_name.lower()

    class TModule(BaseModule, config_class):
        type: t.Literal[module_name]

        def expand_nodes(self) -> t.List["PEDNode"]:
            nonlocal functions, requires_injection
            nodes = []
            for inject, func in zip(requires_injection, functions):
                node = PEDNode.from_callable(
                    func, 
                    static_kwargs={"config": self} if inject else None
                )
                nodes.append(node)
            return nodes
    return TModule

def generate_from_module(module_name: str, module: ModuleType) -> t.Type[BaseModule]:
    """Dynamically generate a BaseModule subclass from all the functions in a given module."""
    functions = [getattr(module, attr) for attr in dir(module) if callable(getattr(module, attr))]
    return generate_from_functions(module_name, *functions)