import typing
import pandas as pd
import inspect
from dataclasses import dataclass, field
from functools import partial


TOutputType = typing.Union[typing.Callable[...,pd.DataFrame], pd.DataFrame, str]

@dataclass
class ConditionedNode:
    class _Unset: ...
    value: typing.Optional[TOutputType] = None
    condition: typing.Any = _Unset
    params: typing.Dict[str,typing.Any] = field(default_factory=dict)


@dataclass
class ChildTree:
    root_tree: "Tree"
    nodes: typing.List[ConditionedNode] = field(default_factory=list)
    default_value: typing.Optional[TOutputType] = None


class Tree:
    __backend_modules__ = [None, "sympy", "numpy"]

    @classmethod
    def from_config(cls, namespace, key):
        # TODO
        raise NotImplemented()
        

    def __init__(self, backend=None) -> None:
        self.root = ChildTree(self)
        self._length = None
        if backend is None:
            from ..settings import settings
            backend = settings.backend
        assert backend in self.__backend_modules__, \
            f"Backend {backend} is invalid. Must be one of {list(self.__backend_modules__.keys())}."
        self._backend = backend

    def _validate_condition(self, condition):
        # TODO
        raise NotImplemented()

    def _validate_output(self, output: TOutputType):
        # TODO this can be done in a more modular way
        if isinstance(output, str):
            return
        if isinstance(output, typing.Callable):
            if not hasattr(output, "__name__"):
                raise ValueError("Callable output must have a __name__ attribute")
            try:
                return_type = inspect.signature(output).return_annotation
            except TypeError as e:
                raise ValueError("Output type must have a valid signature") from e
            if return_type != pd.DataFrame:
                raise ValueError("Return type of callable output must be a pd.DataFrame")
        if isinstance(output, pd.DataFrame):
            output_len = len(output)
            if output_len == 1: return
            if self._length is not None and output_len != self._length:
                raise ValueError(f"All output DataFrames must either have a length of 1 or the same length ({self._length} != {output_len}).")
            # TODO Might cause issues setting it here
            self._length = output_len
            return
        raise ValueError(f"Output is an invalid type. Expecting either a pd.Dataframe, a function or a string.")


    def _add_condition_to_child(self, child: ConditionedNode):
        def inner(output=ConditionedNode._Unset, condition=ConditionedNode._Unset):
            assert (
                child.value is ConditionedNode._Unset or 
                isinstance(child.value, ChildTree)
            ), f"Can only add conditions to a node if no value is set but found {child.value}"
            if child.value is ConditionedNode._Unset:
                child.value = ChildTree(root_tree=self)
            return self.condition(output=output, condition=condition, child_tree=child.value)
        return inner
        

    def _wrap_function(self, condition, child_node: ConditionedNode):
        # Currently only wrap functions can maybe consider wrapping callable
        if not inspect.isfunction(condition): return condition
        condition.condition = self._add_condition_to_child(child_node)
        condition.set_default = partial(
            self.set_default,
            child_tree = child_node.value
        )
        return condition


    def condition(self, output=ConditionedNode._Unset, condition=ConditionedNode._Unset, child_tree: ChildTree=None, **kwargs):
        if child_tree is None:
            child_tree == self.root

        def wrapper(condition):
            node = ConditionedNode(
                value = output,
                condition = condition,
                params = kwargs
            )
            self._validate_output(output)
            child_tree.nodes.append(node)
            return self._wrap_function(condition, child_tree)

        # Allow this to be used as both a wrapper and not
        if condition is not ConditionedNode._Unset:
            return wrapper(condition)
        return wrapper
    
    def set_default(self, output, child_tree: ChildTree=None):
        if child_tree is None:
            child_tree == self.root
        if child_tree.default_value is not None:
            raise ValueError("Default value already set")
        self._validate_output(output)
        child_tree.default_value = output

    def include_subtree(condition=ConditionedNode._Unset):
        # TODO
        raise NotImplemented()
