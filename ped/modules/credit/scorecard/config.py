import typing as t
import polars as pl
from pydantic import Discriminator, Tag, model_validator, PrivateAttr, field_validator
from decider.dag.expanders.base import ConfigurableDeciderExpandableModule
from decider.dag.util import create_node_with_mapping
from .impl import (
    BoundBin, 
    ValuesBin, 
    DefaultBin,
    score_variable,
    default_get_value_from_struct,
    adjust_score,
    calculate_score,
    calculate_probability_of_default,
    log_odds_from_score,
    calculate_credit_score,
)
from decider.serializable import DefinedFunction

def get_bin_type(bin_obj: t.Union[BoundBin, ValuesBin, dict]) -> str:
    if isinstance(bin_obj, dict):
        return "values" if "items" in bin_obj else "bound"
    return "values" if hasattr(bin_obj, 'items') else "bound"

_TBin = t.Annotated[
    t.Union[
        t.Annotated[BoundBin, Tag("bound")],
        t.Annotated[ValuesBin, Tag("values")]
    ],
    Discriminator(get_bin_type)
]


class ScoredVariable(ConfigurableDeciderExpandableModule):
    typ: t.Literal["scored"] = "scored"
    variable_name: str
    bins: t.List[_TBin]
    default: DefaultBin
    strict: bool = True
    raw_output_name: t.Optional[str] = None
    value_output_name: t.Optional[str] = "{variable_name}_score"

    variable_struct_function: t.Optional[DefinedFunction] = None
    struct_to_score_function: t.Optional[DefinedFunction] = None

    _bound_bins: t.List[BoundBin] = PrivateAttr(default_factory=list)
    _value_bins: t.List[ValuesBin] = PrivateAttr(default_factory=list)

    @model_validator(mode='after')
    def validate_bins(self):
        """
        Validates the bins configuration to ensure:
        1. No duplicate values across ValuesBins
        2. BoundBins are in ascending order without overlaps
        3. Proper boundary continuity between consecutive BoundBins
        4. Gap handling based on strict mode setting
        
        In strict mode: consecutive BoundBins must have exact boundary matches (no gaps)
        In non-strict mode: gaps are allowed and will use default values
        
        Examples:
            Valid configurations:
            - [BoundBin(0, 1), BoundBin(1, 2)] (strict=True, continuous)
            - [BoundBin(0, 0.5), BoundBin(1, 2)] (strict=False, gap 0.5-1 uses default)
            - [BoundBin(None, 1), BoundBin(1, None)] (first=-inf to 1, second=1 to +inf)
            - [ValuesBin({'A', 'B'}), BoundBin(0, 1)] (mixed types allowed)
            
            Invalid configurations:
            - [BoundBin(1, 2), BoundBin(0, 1)] (wrong order)
            - [BoundBin(0, 1.5), BoundBin(1, 2)] (overlap)
            - [ValuesBin({'A'}), ValuesBin({'A'})] (duplicate value 'A')
            - [BoundBin(0, None), BoundBin(None, 2)] (unbounded gap)
        
        Raises:
            ValueError: If validation fails with detailed error messages
        """
        if not self.bins:
            return self
        
        # Track all values across ValuesBins to check for duplicates
        all_values = set()
        
        # Track bounds continuity and ordering for BoundBins
        last_bin: t.Optional[BoundBin] = None
        last_bin_idx = -1
        highest_bound = float('-inf')  # Tracks the highest bound seen so far
        _bound_bins = []
        _value_bins = []
        
        for i, bin_obj in enumerate(self.bins):
            if isinstance(bin_obj, ValuesBin):
                # Check for duplicate values across all ValuesBins using set intersection
                _value_bins.append(bin_obj)
                intersection = set(bin_obj.items) & all_values
                if intersection:
                    raise ValueError(f"Duplicate values {intersection} found in ValuesBin at index {i}")
                all_values.update(bin_obj.items)
            
            elif isinstance(bin_obj, BoundBin):
                # Validate that lower_bound < upper_bound when both are defined
                _bound_bins.append(bin_obj)
                if bin_obj.lower_bound is not None and bin_obj.upper_bound is not None:
                    if bin_obj.lower_bound >= bin_obj.upper_bound:
                        raise ValueError(f"BoundBin at index {i} has lower_bound >= upper_bound")
                
                # Check bin ordering: ensure bins are in ascending order
                # First check lower_bound against highest_bound seen so far
                if bin_obj.lower_bound is not None:
                    if bin_obj.lower_bound < highest_bound:
                        raise ValueError(f"BoundBin at index {i} has lower_bound {bin_obj.lower_bound} which is less than previous highest_bound {highest_bound}. Bound bins must be in order and non-overlapping.")
                    highest_bound = bin_obj.lower_bound  # Update to current lower_bound

                # Then check upper_bound - since lower < upper (validated above), 
                # this will effectively set highest_bound to max(lower, upper)
                if bin_obj.upper_bound is not None:
                    if bin_obj.upper_bound < highest_bound:
                        raise ValueError(f"BoundBin at index {i} has upper_bound {bin_obj.upper_bound} which is less than previous highest_bound {highest_bound}. Bound bins must be in order and non-overlapping.")
                    highest_bound = bin_obj.upper_bound  # Update to current upper_bound (the max)

                # Check boundary continuity between consecutive BoundBins
                if last_bin is not None:
                    # If both boundaries are defined, check for proper continuity/gaps
                    if last_bin.upper_bound is not None and bin_obj.lower_bound is not None:
                        # In strict mode: must be exactly continuous (no gaps)
                        # In non-strict mode: allow gaps but prevent overlaps
                        if (self.strict and bin_obj.lower_bound != last_bin.upper_bound) or bin_obj.lower_bound < last_bin.upper_bound:
                            raise ValueError(f"BoundBin at index {i} lower_bound ({bin_obj.lower_bound}) must be {'equal to (in strict mode)' if self.strict else 'greater than or equal to'} previous upper_bound ({last_bin.upper_bound}) defined in BoundBin at index {last_bin_idx}. Bound bins must be in order and non-overlapping.")
                    
                    # Ensure at least one boundary is defined between consecutive bins
                    # This prevents unbounded gaps (e.g., both upper_bound=None and lower_bound=None)
                    if last_bin.upper_bound is None and bin_obj.lower_bound is None:
                        raise ValueError(f"Either BoundBin at index {last_bin_idx} must define a upper_bound or BoundBin at index {i} must have a lower_bound.")
                
                # Update tracking variables for next iteration
                last_bin = bin_obj
                last_bin_idx = i
        self._bound_bins = _bound_bins
        self._value_bins = _value_bins
        return self

    def get_value_output_name(self) -> t.Optional[str]:
        if self.value_output_name is None:
            return None
        return self.value_output_name.format(variable_name=self.variable_name)

    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        nodes = {}
        
        if self.raw_output_name is not None:
            # Create node for score_variable with parameter mapping and partial application
            nodes[self.raw_output_name] = create_node_with_mapping(
                score_variable,
                name=self.raw_output_name,
                input_mapping={self.variable_name: "input"},
                partial_kwargs={
                    "bound_bins": self._bound_bins,
                    "value_bins": self._value_bins,
                    "default_bin": self.default,
                    "input_name": self.variable_name,
                    "output_expr_fn": self.variable_struct_function.get_function() if self.variable_struct_function is not None else None
                }
            )
        
        value_output_name = self.get_value_output_name()
        if value_output_name is not None:
            # Create node for struct-to-value conversion
            value_func = (self.struct_to_score_function.get_function() 
                         if self.struct_to_score_function is not None 
                         else default_get_value_from_struct)
            
            nodes[value_output_name] = create_node_with_mapping(
                value_func,
                name=value_output_name,
                input_mapping={self.raw_output_name: "struct"}
            )
        
        return nodes
    
    def compile(self):
        raise NotImplementedError("Compile method not implemented yet for ScoredVariable")


class AdjustedVariable(ConfigurableDeciderExpandableModule):
    typ: t.Literal["adjusted"] = "adjusted"
    variable: ScoredVariable
    offset: float = 0.0
    scale: float = 1.0
    variable_output_name: str = "{variable_name}_adjusted_score"

    @field_validator("variable", mode="after")
    @classmethod
    def validate_variable(cls, variable: ScoredVariable) -> ScoredVariable:
        if variable.value_output_name is None:
            raise ValueError("ScoredVariable used in AdjustedVariable must have a value_output_name defined to be referenced for adjustment.")
        return variable

    def get_value_output_name(self) -> t.Optional[str]:
        return self.variable_output_name.format(
            variable_name=self.variable.variable_name, 
            score_value_output_name=self.variable.get_value_output_name(),
        )
    
    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        output_name = self.get_value_output_name()
        return {
            output_name: create_node_with_mapping(
                adjust_score,
                name=output_name,
                input_mapping={self.variable.get_value_output_name(): "score"},
                partial_kwargs={"offset": self.offset, "scale": self.scale}
            )
        }
    
class ConstantScore(ConfigurableDeciderExpandableModule):
    typ: t.Literal["constant"] = "constant"
    score: float
    output_name: str = "constant_score"

    def get_value_output_name(self) -> str:
        return self.output_name

    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        def constant_score_fn() -> pl.Expr:
            return pl.lit(self.score)
        
        return {
            self.output_name: create_node_with_mapping(constant_score_fn, name=self.output_name)
        }

_TScoredVariable = t.Annotated[
    t.Union[ScoredVariable, AdjustedVariable, ConstantScore],
    Discriminator("typ")
]

class ScoreCard(ConfigurableDeciderExpandableModule):
    variables: t.List[_TScoredVariable]
    score_output_name: str = "score"

    @field_validator("variables", mode="after")
    @classmethod
    def validate_variables(cls, variables: t.List[_TScoredVariable]) -> t.List[_TScoredVariable]:
        variable_names = set()
        for var in variables:
            if var.typ == "constant": continue
            if var.variable_name in variable_names:
                raise ValueError(f"Duplicate variable_name '{var.variable_name}' found in ScoreCard variables. Each ScoredVariable must have a unique variable_name.")
            variable_names.add(var.variable_name)
            if var.get_value_output_name() is None:
                raise ValueError(f"ScoredVariable with variable_name '{var.variable_name}' must have a value_output_name defined to be used in ScoreCard.")
        return variables
    
    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        # Create nodes for each variable and then a final node that sums them up for the total score
        nodes = {}
        for variable in self.variables:
            nodes.update(variable.expand_nodes(config))
        
        # Get all value output names to create input mapping for calculate_score
        input_mapping = {var_name: var_name for var in self.variables if (var_name :=var.get_value_output_name())}  # Map each output name to itself
        
        nodes[self.score_output_name] = create_node_with_mapping(
            calculate_score,
            input_mapping=input_mapping,
            name=self.score_output_name
        )
        
        return nodes

    def compile(self):
        raise NotImplementedError("Compile method not implemented yet for DecisionTable")
    

class ProbabilityDefault(ConfigurableDeciderExpandableModule):
    typ: t.Literal["probability_default"] = "probability_default"
    score_input_name: str
    probability_output_name: str = "{score_name}_pd"

    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        output_name = self.probability_output_name.format(score_name=self.score_input_name)
        return {
            output_name: create_node_with_mapping(
                calculate_probability_of_default,
                input_mapping={self.score_input_name: "score"},
                name=output_name
            )
        }
    
class LogProbability(ConfigurableDeciderExpandableModule):
    typ: t.Literal["log_probability"] = "log_probability"
    score_input_name: str
    log_probability_output_name: str = "{score_name}_log_odds"

    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        output_name = self.log_probability_output_name.format(score_name=self.score_input_name)
        return {
            output_name: create_node_with_mapping(
                log_odds_from_score,
                input_mapping={self.score_input_name: "score"},
                partial_kwargs={"anchor_score": 660, "target_odds": 15, "points_to_double_the_odds": 20},
                name=output_name
            )
        }
    
class ScoreFromPDO(ConfigurableDeciderExpandableModule):
    typ: t.Literal["score_from_pdo"] = "score_from_pdo"
    pd_input_name: str
    score_output_name: str = "{pd_name}_score"

    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        output_name = self.score_output_name.format(pd_name=self.pd_input_name)
        return {
            output_name: create_node_with_mapping(
                calculate_credit_score,
                input_mapping={self.pd_input_name: "probability_of_default"},
                name=output_name
            )
        }


class WeightedScore(t.NamedTuple):
    score_name: str
    weight: float

class MergeScorecardValues(ConfigurableDeciderExpandableModule):
    typ: t.Literal["merge_scorecard_values"] = "merge_scorecard_values"
    weighted_scores: t.List[WeightedScore]
    output_name: str = "merged_scorecard_values"

    @field_validator("weighted_scores", mode="after")
    @classmethod
    def validate_weighted_scores(cls, weighted_scores: t.List[WeightedScore]) -> t.List[WeightedScore]:
        total_weight = sum(ws.weight for ws in weighted_scores)
        if total_weight != 1.0:
            raise ValueError(f"The sum of weights in weighted_scores must equal 1.0. Current sum is {total_weight}.")
        return weighted_scores

    def expand_nodes(self) -> t.Dict[str, t.Any]:
        # TODO i just made this up i think its more complex than this as pd is involved here. @christiaan
        def merge_scorecard_values(**kwargs):
            # Create a weighted sum of the scores
            result = 0.0
            for weighted_score in self.weighted_scores:
                score_value = kwargs[weighted_score.score_name]
                result += score_value * weighted_score.weight
            return result
        
        # Create input mapping for all the weighted scores
        input_mapping = {ws.score_name: ws.score_name for ws in self.weighted_scores}
        
        return {
            self.output_name: create_node_with_mapping(
                merge_scorecard_values,
                input_mapping=input_mapping
            )
        }