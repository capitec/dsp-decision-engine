import typing
import collections.abc
from functools import partial
from dataclasses import dataclass
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, ConfigDict
from spockflow.nodes import VariableNode


TOutput = typing.TypeVar("TOutput")
TCond = typing.TypeVar("TCond")
TNodeType = typing.TypeVar("TNodeType")

class ConditionedNode(BaseModel, typing.Generic[TOutput,TCond]):
    # TODO fix for pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: typing.Union[TOutput,"ChildTree",None] = None
    condition: typing.Optional[TCond] = None

    @staticmethod
    def _length_attr(attr):
        if attr is None: return 1
        if isinstance(attr, str): return 1
        if not isinstance(attr, collections.abc.Sized): return 1
        return len(attr)
    
    def __len__(self):
        len_attr = self._length_attr(self.value)
        if len_attr == 1:
            len_attr = self._length_attr(self.condition)
        return len_attr
    
    @model_validator(mode='after')
    def check_compatible_lengths(self) -> Self:
        len_value = self._length_attr(self.value)
        if len_value == 1: return self
        len_condition = self._length_attr(self.condition)
        if len_condition == 1: return self
        if len_condition != len_value:
            raise ValueError('Condition and value lengths incompatible')
        return self


class ChildTree(BaseModel, typing.Generic[TOutput,TCond,TNodeType]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nodes: typing.List[TNodeType] = Field(default_factory=list)
    default_value: typing.Optional[TOutput] = None

    @property
    def NodeType(self) -> TNodeType:
        return ConditionedNode[TCond, TOutput]

    def __len__(self): # TODO could maybe cache this on bigger trees if the values havent changed
        node_len = self.NodeType._length_attr(self.default_value)
        if node_len != 1: return node_len
        for node in self.nodes:
            if (node_len:=len(node)) != 1: return node_len
        return 1

    @model_validator(mode='after')
    def check_compatible_lengths(self) -> Self:
        child_tree_len = self.NodeType._length_attr(self.default_value)
        for node in self.nodes:
            len_value = len(node)
            if len_value == 1: continue
            if child_tree_len == 1:
                child_tree_len = len_value
            elif child_tree_len != len_value:
                raise ValueError(f'Lengths of values or conditions in the tree is incompatible. Found {child_tree_len} != {len_value}.')
        return self

    def add_node(self, value: TOutput, condition: TCond, **kwargs) -> TNodeType:
        len_value = self.NodeType._length_attr(value)
        len_condition = self.NodeType._length_attr(condition)
        if len_value != 1 and len_condition != 1 and len_value != len_condition:
            raise ValueError(f"Cannot add node as the length of the value ({len_value}) is not compatible with the length of the condition ({len_condition}).")
        if len_value != 1 or len_condition != 1:
            # TODO adding this allows better validation but requires circular loops so hard for pydantic to serialise
            # len_tree = len(self.root_tree.root)
            len_tree = len(self)
            if len_value != 1 and len_tree != 1 and len_value != len_tree:
                raise ValueError(f"Cannot add node as the length of the value ({len_value}) incompatible with tree {len_tree}.")
            if len_condition != 1 and len_tree != 1 and len_condition != len_tree:
                raise ValueError(f"Cannot add node as the length of the condition ({len_condition}) incompatible with tree {len_tree}.")
        node = self.NodeType(value=value, condition=condition, **kwargs)
        self.nodes.append(node)
        return node


    def set_default(self, value: TOutput):
        if self.default_value is not None:
            raise ValueError("Default value already set")
        len_value = self.NodeType._length_attr(value)
        if len_value != 1:
            # TODO adding this allows better validation but requires circular loops so hard for pydantic to serialise
            # len_tree = len(self.root_tree.root)
            len_tree = len(self)
            if len_tree != 1 and len_value != len_tree:
                raise ValueError(f"Cannot set default as length of value ({len_value}) incompatible with tree {len_tree}.")
            
    def merge_into(self, other: Self):
        len_subtree = len(other)
        if len_subtree != 1:
            len_tree = len(self)
            if len_tree != 1 and len_subtree != len_tree:
                raise ValueError(f"Subtree length ({len_subtree}) incompatible with tree ({len_tree}).")
            
        if other.default_value is not None and self.default_value is not None:
            raise ValueError(f"Cannot merge two subtrees both containing default values")
        self.nodes.extend(other.nodes)

class WrappedTreeFunction(ABC,typing.Generic[TOutput,TCond]):
    @abstractmethod
    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any: ...
    @abstractmethod
    def condition(self, output: TOutput=None, condition: TCond=None, **kwargs) -> typing.Union[Self,TCond]: ...
    @abstractmethod
    def set_default(self, output: TOutput): ...
    @abstractmethod
    def include_subtree(self, subtree: typing.Union[Self, "ChildTree"], condition=None, **kwargs): ...



TChildTree = typing.TypeVar("TChildTree", bound=ChildTree[ConditionedNode[TOutput,TCond],TOutput,TCond])
class Tree(VariableNode, typing.Generic[TChildTree,TOutput,TCond]):
    root: TChildTree = None
    
    @model_validator(mode='before')
    @classmethod
    def default_empty_root(cls, data: typing.Any) -> typing.Any:
        if isinstance(data, dict):
            if 'root' not in data:
                data['root'] = cls.TreeType()()
        return data

    @classmethod
    def TreeType(cls) -> ChildTree:
        return ChildTree[ConditionedNode[TOutput,TCond],TOutput,TCond]

    def _inject_child_tree_from_node(self, *args, child_node: ConditionedNode, function, **kwargs):
        if child_node.value is None:
            child_node.value = self.TreeType()()
        if not isinstance(child_node.value, self.TreeType()):
            raise ValueError(f"Subtree must have no value set or already be associated to a subtree. Found value = {child_node.value}")
        return function(*args, **kwargs, child_tree=child_node.value)
        
    def _wrap_function(self, condition, child_node: ConditionedNode):
        import inspect
        # Currently only wrap functions can maybe consider wrapping callable
        if not inspect.isfunction(condition): 
            def raise_cannot_wrap_error(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], typing.Callable):
                    raise ValueError("Cannot wrap function when condition is specified")
                raise NotImplementedError()
            condition = raise_cannot_wrap_error
        # condition.condition = self._add_condition_to_child(child_node)
        condition.condition = partial(
            self._inject_child_tree_from_node,
            child_node = child_node,
            function = self.condition
        )
        condition.set_default = partial(
            self._inject_child_tree_from_node,
            child_node = child_node,
            function = self.set_default
        )
        condition.include_subtree = partial(
            self._inject_child_tree_from_node,
            child_node = child_node,
            function = self.include_subtree
        )
        return condition
    
    def _identify_loops(self, *nodes: 'ConditionedNode'):
        q = list(nodes)
        seen = set()
        while q:
            el = q.pop()
            if id(el) in seen:
                raise ValueError("Tree must not contain any loops")
            seen.add(id(el))
            if isinstance(el.value, ChildTree):
                q.extend(el.value.nodes)
                if isinstance(el.value.default_value, ChildTree):
                    q.extend(el.value.default_value.nodes)

    @staticmethod
    def _remove_nodes_from_end(*nodes: 'ConditionedNode', child_tree: ChildTree):
        # Not the most pythonic method can maybe be improved
        # Done to remove elements in the order they were added
        nodes_to_remove = list(nodes)
        last_el_idx = len(child_tree.nodes)-1
        for rev_i, node in enumerate(reversed(child_tree.nodes)):
            i = last_el_idx - rev_i
            if node in nodes_to_remove:
                child_tree.nodes.pop(i)
            if len(nodes_to_remove) <= 0: return True
        return False
    
    def copy(self, deep=True):
        return self.__class__(self.root.model_copy(deep=deep))

    # def parameterize_condition(self) # TODO
    def condition(self, output: TOutput=None, condition: TCond=None, child_tree: ChildTree=None, **kwargs) -> typing.Callable[..., WrappedTreeFunction[TOutput, TCond]] :
        if child_tree is None:
            child_tree = self.root

        def wrapper(condition: TCond):
            nonlocal output
            if isinstance(output, Tree):
                output = output.root
            node = child_tree.add_node(
                value = output,
                condition = condition,
                **kwargs
            )
            try:
                self._identify_loops(node)
            except ValueError as e:
                # Try to keep the order
                self._remove_nodes_from_end(node, child_tree=child_tree)
                raise e from e
            return self._wrap_function(condition, node)

        # Allow this to be used as both a wrapper and not
        if condition is not None:
            return wrapper(condition)
        return wrapper
    
    def set_default(self, output: TOutput, child_tree: ChildTree=None):
        if child_tree is None:
            child_tree = self.root
        if child_tree.default_value is not None:
            raise ValueError("Default value already set")
        if isinstance(output, Tree):
            output = output.root
        child_tree.default_value = output

    def include_subtree(self, subtree: typing.Union[Self, ChildTree], condition=None, child_tree: ChildTree=None, **kwargs):
        if child_tree is None:
            child_tree = self.root
        
        if isinstance(subtree, Tree):
            subtree = subtree.root
            
        if condition is None:
            child_tree.merge_into(subtree)
            new_nodes = child_tree.nodes
        else:
            new_nodes = [child_tree.add_node(subtree, condition, **kwargs)]

        try:
            self._identify_loops(*new_nodes)
        except ValueError as e:
            # Try to keep the order
            self._remove_nodes_from_end(*new_nodes, child_tree=child_tree)
            raise e from e
