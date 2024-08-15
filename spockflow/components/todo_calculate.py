import sys
import typing
import inspect
import types
import pandas as pd

from hamilton import node

from ..nodes import SimpleVariableNode, VariableNodeExpander, InputTypes

# TODO potentially make a Calculate Node from a variable Node

T = typing.TypeVar("T")


def calculate(
    func: typing.Callable[[], typing.Any],
    return_type: T = pd.Series,
    doc: typing.Optional[str] = None,
) -> T:
    # Extract the globally referenced function definitions
    func_globals = inspect.getclosurevars(func).globals

    input_types: InputTypes = {}
    for g_name, g_value in func_globals.items():
        if isinstance(g_value, VariableNodeExpander):
            input_types[g_name] = (g_value.return_type, node.DependencyType.REQUIRED)
        elif inspect.isfunction(g_value):
            type_hint_kwargs = (
                {} if sys.version_info < (3, 9) else {"include_extras": True}
            )
            rt = typing.get_type_hints(func, **type_hint_kwargs).get("return")
            if return_type is None:
                raise ValueError(
                    f"Missing type hint for return value in function {fn.__qualname__}."
                )
            input_types[g_name] = (rt, node.DependencyType.REQUIRED)

    def fn(**kwargs):
        enclosed_fn = types.LambdaType(func.__code__, kwargs)
        return enclosed_fn()

    return SimpleVariableNode(
        fn, return_type, doc, inspect.getmodule(func).__name__, input_types=input_types
    )
