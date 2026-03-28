from enum import Enum, auto
import typing as t
from uuid import uuid4
import typing_extensions as t_ext
from pydantic import BaseModel, Field, model_validator, PrivateAttr, field_validator
import re
import pandas as pd
import numpy as np
from .core_node_data import (
    RangeNodeCoreData,
    NumericalNodeCoreData,
    StringMatchNodeCoreData,
    CategoricalNodeCoreData,
    LeafNodeCoreData,
)

if t.TYPE_CHECKING:
    from treelite.model_builder import ModelBuilder
    from .tree import Tree

VARIABLE_NODE_TYPE_SUFFIX = "_compiled_with_variables"


class RangeTestNode(RangeNodeCoreData):
    IS_LEAF: t.ClassVar[bool] = False
    preprocess_inputs: t.ClassVar[None] = None
    # Overshadow this so we can add in extra validation
    thresholds: t.List[float] = Field(min_length=1)

    @property
    def child_count(self):
        return len(self.thresholds) + 1

    @model_validator(mode="after")
    def validate_thresholds(self) -> t_ext.Self:
        if len(self.thresholds) < 1:
            raise ValueError(f"Range test nodes should have at least one threshold")
        return self

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        processed_feature_id: int,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        for t1, t2 in zip(self.thresholds, self.thresholds[1:]):
            assert (
                t2 > t1
            ), f"Thresholds for range node {node_id} must be in strict ascending order. Found {t2} <= {t1}"
        children = node_children.get(node_id, [])
        self._build_balanced_tree(
            builder,
            split_node_id=node_id,
            node_lookup=node_lookup,
            thresholds=self.thresholds,
            children=children[: len(self.thresholds) + 1],
            processed_feature_id=processed_feature_id,
        )

    def _build_balanced_tree(
        self,
        builder: "ModelBuilder",
        split_node_id: str,
        node_lookup: t.Dict[str, int],
        thresholds: t.List[float],
        children: t.List[str],
        processed_feature_id: int,
    ):
        if not len(thresholds):
            return
        # Update the node lookup
        left_node_key = f"{split_node_id}_L"
        if left_node_key in node_lookup:
            left_node_key = f"{left_node_key}_{uuid4()}"
        right_node_key = f"{split_node_id}_R"
        if right_node_key in node_lookup:
            right_node_key = f"{right_node_key}_{uuid4()}"
        node_lookup[left_node_key] = len(node_lookup)
        node_lookup[right_node_key] = len(node_lookup)
        # Perform splits down the middle
        mid_index = len(thresholds) // 2
        builder.start_node(node_lookup[split_node_id])
        builder.numerical_test(
            feature_id=processed_feature_id,
            threshold=thresholds[mid_index],
            default_left=self.default_left,
            opname="<",
            left_child_key=(
                node_lookup[children[0]]
                if len(thresholds) == 1
                else node_lookup[left_node_key]
            ),
            # We do lt or eq 2 here because if there are 2 thresholds [x,y]
            # Then this will be <y and the next two calls will be [x], [] so the left branch will handle
            # The last split and the right branch will directly point to the child
            right_child_key=(
                node_lookup[children[len(children) - 1]]
                if len(thresholds) <= 2
                else node_lookup[right_node_key]
            ),
        )
        builder.end_node()
        self._build_balanced_tree(
            builder,
            split_node_id=left_node_key,
            node_lookup=node_lookup,
            thresholds=thresholds[:mid_index],
            children=children[: mid_index + 1],  # We not leaving a child behind
            processed_feature_id=processed_feature_id,
        )
        self._build_balanced_tree(
            builder,
            split_node_id=right_node_key,
            node_lookup=node_lookup,
            thresholds=thresholds[mid_index + 1 :],
            children=children[mid_index + 1 :],
            processed_feature_id=processed_feature_id,
        )


class NumericalTestNode(NumericalNodeCoreData):
    IS_LEAF: t.ClassVar[bool] = False
    preprocess_inputs: t.ClassVar[None] = None

    @property
    def child_count(self):
        return 2

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        processed_feature_id: int,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        assert processed_feature_id is not None
        children = node_children.get(node_id, [])
        builder.start_node(node_lookup[node_id])
        builder.numerical_test(
            feature_id=processed_feature_id,
            threshold=self.threshold,
            default_left=self.default_left,
            opname=self.comparison_op,
            left_child_key=node_lookup[children[0]] if len(children) > 0 else 0,
            right_child_key=node_lookup[children[1]] if len(children) > 1 else 0,
        )
        builder.end_node()


class NumericalTestNodeWithVariables(NumericalTestNode):
    # This node makes use of x > t -> x - t > 0 (works for all ops)
    _USES_VARIABLES: t.ClassVar[bool] = True
    node_type: t.ClassVar[str] = NumericalTestNode.NODE_TYPE + VARIABLE_NODE_TYPE_SUFFIX
    variable_key: str
    default_variable_value: float

    @model_validator(mode="after")
    def threshold_always_o(self) -> t_ext.Self:
        # The preprocess inputs always ensures the threshold is 0
        self.threshold = 0
        return self

    def preprocess_inputs(
        self, inputs: pd.Series, variables: t.Dict[str, t.Union[str, float, int]]
    ) -> np.ndarray:
        return inputs.astype(float) - float(
            variables.get(self.variable_key, self.default_variable_value)
        )


class CategoricalTestNode(CategoricalNodeCoreData):
    IS_LEAF: t.ClassVar[bool] = False
    preprocess_inputs: t.ClassVar[None] = None

    @property
    def child_count(self):
        return 2

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        processed_feature_id: int,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        assert processed_feature_id is not None
        children = node_children.get(node_id, [])
        builder.start_node(node_lookup[node_id])
        builder.categorical_test(
            feature_id=processed_feature_id,
            default_left=self.default_left,
            left_child_key=node_lookup[children[0]] if len(children) > 0 else 0,
            right_child_key=node_lookup[children[1]] if len(children) > 1 else 0,
            category_list=self.category_list,
            category_list_right_child=self.category_list_right_child,
        )
        builder.end_node()


class CategoricalTestNodeWithVariables(CategoricalTestNode):
    _USES_VARIABLES: t.ClassVar[bool] = True
    node_type: t.ClassVar[str] = (
        CategoricalTestNode.NODE_TYPE + VARIABLE_NODE_TYPE_SUFFIX
    )
    # Using one dict for both the list of items and the defaults
    variable_values: t.Dict[str, int]

    def preprocess_inputs(
        self, inputs: pd.Series, variables: t.Dict[str, t.Union[str, float, int]]
    ) -> np.ndarray:
        item_set = {
            int(variables.get(k, default_value))
            for k, default_value in self.variable_values.items()
        }.union(self.category_list)
        return inputs.astype(int).isin(item_set).astype(int)

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        processed_feature_id: int,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        assert processed_feature_id is not None
        children = node_children.get(node_id, [])
        builder.start_node(node_lookup[node_id])
        builder.numerical_test(
            feature_id=processed_feature_id,
            threshold=0,
            default_left=self.default_left,
            # Numerical tests go left if correct
            # 1 if is in 0 if not
            # > will go to left if is in
            # <= will go right if is in
            opname="<=" if self.category_list_right_child else ">",
            left_child_key=node_lookup[children[0]] if len(children) > 0 else 0,
            right_child_key=node_lookup[children[1]] if len(children) > 1 else 0,
        )
        builder.end_node()


class StringMatchNode(StringMatchNodeCoreData):
    IS_LEAF: t.ClassVar[bool] = False
    _compiled_pattern: re.Pattern = PrivateAttr(default=None)
    patterns: t.List[str] = Field(min_length=1)

    @classmethod
    def _contains_preprocessor(cls, val: str):
        return f"^.*{re.escape(val)}.*$"

    @classmethod
    def _starts_with_preprocessor(cls, val: str):
        return f"^{re.escape(val)}.*$"

    @classmethod
    def _ends_with_preprocessor(cls, val: str):
        return f"^.*{re.escape(val)}$"

    @classmethod
    def _exact_preprocessor(cls, val: str):
        return f"^{re.escape(val)}$"

    @property
    def _pattern_text_preprocessor(self):
        if self.match_type == "regex":
            return lambda x: x
        elif self.match_type == "starts_with":
            return self._starts_with_preprocessor
        elif self.match_type == "contains":
            return self._contains_preprocessor
        elif self.match_type == "ends_with":
            return self._ends_with_preprocessor
        elif self.match_type == "exact":
            return self._exact_preprocessor
        raise ValueError("Invalid Match Type")

    @property
    def child_count(self):
        return 2 if self.match_any else (len(self.patterns) + 1)

    def _validate_patterns(self, patterns: t.List[str]):
        pre_proc = self._pattern_text_preprocessor
        pattern_parts = []
        for pattern in patterns:
            try:
                pre_proc_pattern = pre_proc(pattern)
                compiled_pattern = re.compile(pre_proc_pattern)
                if compiled_pattern.groups > 0:
                    raise ValueError(
                        f"Pattern '{pre_proc_pattern}' cannot contain capture groups. Please use (?:) if a group is required."
                    )
                pattern_parts.append(f"({pre_proc_pattern})")
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{e.pattern}': {e}")
        pattern_parts.append("(.*)")  # Final catch all
        return re.compile(
            "|".join(pattern_parts),  # make one big pattern,
            flags=0 if self.case_sensitive else re.IGNORECASE,
        )

    @model_validator(mode="after")
    def validate_patterns(self) -> t_ext.Self:
        # TODO we must probably update this everytime the patterns update.
        self._compiled_pattern = self._validate_patterns(self.patterns)
        return self

    def preprocess_inputs(
        self, inputs: pd.Series, variables: t.Dict[str, t.Union[str, float, int]] = None
    ) -> np.ndarray:
        # We use capture groups to determine which pattern is matched
        # inputs = pd.Series(["horse", "eagle", "donkey"])
        # _compiled_pattern = "(^.*h.*)|(e)|(.*)"
        # inputs.str.extract(_compiled_pattern)
        # horse	NaN	NaN
        # NaN	e	NaN
        # NaN	NaN	donkey
        # ~inputs.str.extract(_compiled_pattern).isna()
        # True	False	False
        # False	True	False
        # False	False	True
        # Determine index of first non nan and we have our pattern index
        values = np.argmax(
            ~inputs.astype(str).str.extract(self._compiled_pattern).isna(), axis=1
        ).astype(np.float64)
        # otherwise nantypes make their way through as index 0
        values[inputs.isna()] = np.nan
        return values

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        processed_feature_id: int,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        assert processed_feature_id is not None
        children = node_children.get(node_id, [])
        if self.match_any:
            builder.start_node(node_lookup[node_id])

            builder.numerical_test(
                feature_id=processed_feature_id,
                threshold=len(
                    self.patterns
                ),  # no match will go to end of list (.* catch all pattern)
                default_left=self.default_left,
                opname=">=",
                left_child_key=node_lookup[children[1]],
                right_child_key=node_lookup[children[0]],
            )
            builder.end_node()
            return
        # Cant balance this one have to build a linear tree
        #        p0
        #       / \ (right means its a match)
        #      p1  o1
        #     / \
        #    p2  o2
        #   / \
        #  o4  o3
        # Correction we should be able to balance this one as the argmax should already deal with priority
        # That can be left as something to do in the future as an optimization to take this from O(P) to O(log2(P))
        node_keys = [node_id]  # Keep the first split no need to make a new node id
        for i in range(1, len(self.patterns)):
            node_key = f"{node_keys}_{i}"
            if node_key in node_lookup:
                node_key = f"{node_keys}_{uuid4()}"
            node_keys.append(node_key)
            node_lookup[node_key] = len(node_lookup)

        for pat_i, (node_key, next_node_key, child) in enumerate(
            zip(node_keys, node_keys[1:], children)
        ):
            builder.start_node(node_lookup[node_key])
            builder.categorical_test(
                feature_id=processed_feature_id,
                default_left=True,
                left_child_key=node_lookup[next_node_key],
                right_child_key=node_lookup[child],
                category_list=[pat_i],
                category_list_right_child=True,  # if value in list it should go right
            )
            builder.end_node()

        builder.start_node(node_lookup[node_keys[-1]])
        builder.categorical_test(
            feature_id=processed_feature_id,
            default_left=True,
            left_child_key=node_lookup[children[-1]],
            right_child_key=node_lookup[children[-2]],
            category_list=[len(self.patterns) - 1],
            category_list_right_child=True,  # if value in list it should go right
        )
        builder.end_node()


class StringMatchNodeWithVariables(StringMatchNode):
    _USES_VARIABLES: t.ClassVar[bool] = True
    # Using one dict for both the list of items and the defaults
    variable_values: t.Dict[str, str]

    @field_validator("match_any", mode="after")
    @classmethod
    def match_any_always_true(cls, value):
        if not value:
            raise ValueError(
                "String match node does not support variables with match_any=False"
            )
        return value

    @model_validator(mode="after")
    def validate_patterns(self) -> t_ext.Self:
        # This overrides the default. We hoping here that the variables dont change much so we can reuse the default pattern
        self._compiled_pattern = self._validate_patterns(
            self.patterns + list(self.variable_values.values())
        )
        return self

    def preprocess_inputs(
        self, inputs: pd.Series, variables: t.Dict[str, str]
    ) -> np.ndarray:

        extra_values = []
        needs_compile = False
        for k, default_value in self.variable_values.items():
            nv = str(variables.get(k, default_value))
            extra_values.append(nv)
            if not needs_compile and nv != default_value:
                # TODO see if its quicker doing the compile or the check to see if we need the compile
                needs_compile = True

        compiled_pattern = self._compiled_pattern
        if needs_compile:
            compiled_pattern = self._validate_patterns(self.patterns + extra_values)

        # Could probably use the simpler match logic
        values = (
            np.argmax(
                ~inputs.astype(str).str.extract(compiled_pattern).isna(), axis=1
            ).astype(np.float64)
            == self._compiled_pattern.groups - 1
        )
        # otherwise nantypes make their way through as index 0
        values[inputs.isna()] = np.nan
        return values

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        processed_feature_id: int,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        assert processed_feature_id is not None
        children = node_children.get(node_id, [])

        builder.start_node(node_lookup[node_id])

        builder.numerical_test(
            feature_id=processed_feature_id,
            threshold=0,  # no match will be 1, match will be 0 due to last capture group being the catch all ().*)
            default_left=self.default_left,
            opname="<=",
            left_child_key=node_lookup[children[1]],
            right_child_key=node_lookup[children[0]],
        )
        builder.end_node()
        return


class OutputEncoding(str, Enum):
    ONE_HOT = auto()
    INDEX = auto()


def _one_hot(n_classes: int, class_index: int):
    """Generates a one hot encoding note that here the class index ranges from -1 to n_classes-1

    Args:
        n_classes (int): The number of classes C
        class_index (int): the index in range [-1, C-1)

    Returns:
        t.List[int]: A one hot encoded list
    """
    return [int(i - 1 == class_index) for i in range(n_classes)]


class LeafNode(LeafNodeCoreData):
    IS_LEAF: t.ClassVar[bool] = True
    DEFAULT_LEAF_VALUE: t.ClassVar[str] = -1

    @property
    def child_count(self):
        return 0

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: t.Dict[str, int],
        output_encoding: OutputEncoding,
        node_children: t.Dict[str, t.List[str]],
        tree: "Tree" = None,
        **kwargs,
    ):
        builder.start_node(node_lookup[node_id])
        builder.leaf(
            [node_lookup[node_id]]
            if output_encoding == OutputEncoding.INDEX
            else _one_hot(len(tree.tree_output.data), self.leaf_value)
        )
        builder.end_node()


CompiledNodeType = t_ext.Annotated[
    t.Union[
        RangeTestNode,
        NumericalTestNode,
        CategoricalTestNode,
        StringMatchNode,
        LeafNode,
        NumericalTestNodeWithVariables,
        CategoricalTestNodeWithVariables,
        StringMatchNodeWithVariables,
    ],
    Field(discriminator="node_type"),
]
