import sys
import typing
import inspect
import types
import pandas as pd
from spockflow.nodes import VariableNode, creates_node
from hamilton import node


T = typing.TypeVar("T")


class Calculate(VariableNode, typing.Generic[T]):
    func: typing.Callable[[], typing.Any]
    return_type: typing.Type[T] = pd.Series
    doc: typing.Optional[str] = None

    def _generate_nodes(
        self,
        name: str,
        config: "typing.Dict[str, typing.Any]",
        include_runtime_nodes: bool = False,
    ) -> "typing.List[node.Node]":

        func_globals = inspect.getclosurevars(self.func).globals

        input_types = {}
        for g_name, g_value in func_globals.items():
            if isinstance(g_value, VariableNodeExpander):
                input_types[g_name] = (typing.Any, node.DependencyType.REQUIRED)
            elif inspect.isfunction(g_value):
                type_hint_kwargs = (
                    {} if sys.version_info < (3, 9) else {"include_extras": True}
                )
                rt = typing.get_type_hints(self.func, **type_hint_kwargs).get("return")
                if rt is None:
                    raise ValueError(
                        f"Missing type hint for return value in function {fn.__qualname__}."
                    )
                input_types[g_name] = (rt, node.DependencyType.REQUIRED)

        def fn(**kwargs):
            enclosed_fn = types.LambdaType(self.func.__code__, kwargs)
            return enclosed_fn()

        return [
            node.Node(
                name=name,
                typ=self.return_type,
                doc_string=self.doc,
                callabl=fn,
                input_types=input_types,
            )
        ]
