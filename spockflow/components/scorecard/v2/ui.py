import traitlets
import ipywidgets as widgets
from IPython.display import display
import json
import sys

import typing

from spockflow.components.scorecard.v2.common import (
    DefaultScorePattern,
    RangeScorePattern,
)
from spockflow.components.scorecard.v2.criteria_categorical import (
    ScoreCriteriaCategorical,
    CategoricalDiscreteScorePattern,
)
from spockflow.components.scorecard.v2.criteria_numerical import (
    ScoreCriteriaNumerical,
    Bounds,
    NumericalDiscreteScorePattern,
)
from spockflow.components.scorecard.v2.scorecard import ScoreCardModel


class DiscreteScorePatternViewModel(traitlets.HasTraits):
    values = traitlets.List(traitlets.Unicode())
    group_id = traitlets.Int(0)
    score = traitlets.Float(0)
    description = traitlets.Unicode("")
    null_value = traitlets.Bool(False)

    @staticmethod
    def from_pydantic_model(
        model: "CategoricalDiscreteScorePattern[typing.Optional[str]]",
    ):
        values = [v for v in model.values if v is not None]
        has_none = len(values) < len(model.values)
        return DiscreteScorePatternViewModel(
            values=values,
            group_id=model.group_id,
            score=model.score,
            description=model.description,
            null_value=has_none,
        )

    def to_pydantic_model(self):
        extra_values = [None] if self.null_value else []
        return CategoricalDiscreteScorePattern(
            values=self.values + extra_values,
            group_id=self.group_id,
            score=self.score,
            description=self.description,
        )

    def get_widget(self, should_display=True, on_delete_callback=None):
        values = widgets.TagsInput(
            value=self.values,
            allow_duplicates=False,
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((values, "value"), (self, "values"))

        score = widgets.FloatText(
            value=self.score, description="Score:", layout=widgets.Layout(width="auto")
        )
        widgets.link((score, "value"), (self, "score"))

        score_description = widgets.Textarea(
            value=self.description,
            description="Description:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_description, "value"), (self, "description"))

        score_idx = widgets.IntText(
            value=self.group_id,
            description="Class ID:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_idx, "value"), (self, "group_id"))

        delete_row = widgets.Button(
            # description='',
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(
                right="5px",
                top="5px",
                width="32px",
                height="28px",
            ),
        )
        if on_delete_callback is not None:
            delete_row.on_click(lambda evnt: on_delete_callback(self))

        null_value = widgets.ToggleButton(
            value=self.null_value,
            description="Null",
            layout=widgets.Layout(width="5em"),
        )
        widgets.link((null_value, "value"), (self, "null_value"))

        # def on_click(change):
        #     print(change['new'])

        # null_value.observe(on_click, 'value')

        categorical_row_config_grid = widgets.GridspecLayout(
            1, 6, layout=widgets.Layout(width="100%", grid_gap="5px")
        )
        categorical_row_config_grid[0, 0] = score
        categorical_row_config_grid[0, 1] = score_idx
        categorical_row_config_grid[0, 2:4] = score_description
        categorical_row_config_grid[0, 4:] = widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.Label(
                            value="Values: ", layout=widgets.Layout(width="6em")
                        ),
                        null_value,
                    ]
                ),
                values,
            ],
            layout=widgets.Layout(display="grid", grid_template_columns="auto 1fr"),
        )
        categorical_row_config = widgets.HBox(
            [categorical_row_config_grid, delete_row],
            layout=widgets.Layout(width="100%", border="1px solid grey"),
        )

        if should_display:
            display(categorical_row_config)
        return categorical_row_config


class DiscreteFloatScorePatternViewModel(traitlets.HasTraits):
    # Really far from ideal to leave this as unicode and do post validation
    # Cant think of a much better way for now than to make a custom component
    values = traitlets.List(traitlets.Unicode())
    group_id = traitlets.Int(0)
    score = traitlets.Float(0)
    description = traitlets.Unicode("")

    @staticmethod
    def from_pydantic_model(model: "NumericalDiscreteScorePattern[float]"):
        return DiscreteFloatScorePatternViewModel(
            values=[str(v) for v in model.values],
            group_id=model.group_id,
            score=model.score,
            description=model.description,
        )

    @staticmethod
    def convert_float(value: str):
        if isinstance(value, float):
            return value
        lower_val = value.lower()
        if lower_val in ["none", "null"]:
            return None
        if lower_val == "inf":
            return float("inf")
        if lower_val == "-inf":
            return -float("inf")
        return float(value)

    def to_pydantic_model(self):
        return NumericalDiscreteScorePattern(
            values=[self.convert_float(v) for v in self.values],
            group_id=self.group_id,
            score=self.score,
            description=self.description,
        )

    def get_widget(self, should_display=True, on_delete_callback=None):
        values = widgets.TagsInput(
            value=self.values,
            allow_duplicates=False,
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((values, "value"), (self, "values"))

        score = widgets.FloatText(
            value=self.score, description="Score:", layout=widgets.Layout(width="auto")
        )
        widgets.link((score, "value"), (self, "score"))

        score_description = widgets.Textarea(
            value=self.description,
            description="Description:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_description, "value"), (self, "description"))

        score_idx = widgets.IntText(
            value=self.group_id,
            description="Class ID:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_idx, "value"), (self, "group_id"))

        delete_row = widgets.Button(
            # description='',
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(
                width="32px",
                height="28px",
            ),
        )
        if on_delete_callback is not None:
            delete_row.on_click(lambda evnt: on_delete_callback(self))

        categorical_row_config_grid = widgets.GridspecLayout(
            1, 6, layout=widgets.Layout(width="100%", grid_gap="5px")
        )
        categorical_row_config_grid[0, 0] = score
        categorical_row_config_grid[0, 1] = score_idx
        categorical_row_config_grid[0, 2:4] = score_description
        categorical_row_config_grid[0, 4:] = widgets.HBox(
            [
                widgets.Label(value="Values: ", layout=widgets.Layout(width="6em")),
                values,
            ],
            layout=widgets.Layout(display="grid", grid_template_columns="auto 1fr"),
        )

        # categorical_row_config_grid[0,4:]= widgets.HBox([widgets.VBox([widgets.Label(value="Values: ", layout=widgets.Layout(width='6em')), null_value]), values], layout=widgets.Layout(display='grid', grid_template_columns='auto 1fr'))
        categorical_row_config = widgets.HBox(
            [categorical_row_config_grid, delete_row],
            layout=widgets.Layout(width="100%", border="1px solid grey"),
        )

        if should_display:
            display(categorical_row_config)
        return categorical_row_config


class DefaultScorePatternViewModel(traitlets.HasTraits):
    group_id = traitlets.Int(0)
    score = traitlets.Float(0)
    description = traitlets.Unicode("")
    is_active = traitlets.Bool(False)

    @staticmethod
    def from_pydantic_model(
        model: "typing.Optional[DefaultScorePattern]",
    ) -> "DefaultScorePatternViewModel":
        if model is None:
            return DefaultScorePatternViewModel(is_active=False)
        return DefaultScorePatternViewModel(
            group_id=model.group_id,
            score=model.score,
            description=model.description,
            is_active=True,
        )

    def to_pydantic_model(self) -> DefaultScorePattern:
        if not self.is_active:
            return None
        return DefaultScorePattern(
            group_id=self.group_id,
            score=self.score,
            description=self.description,
        )

    def get_widget(self, should_display=True):
        score = widgets.FloatText(
            value=self.score, description="Score:", layout=widgets.Layout(width="auto")
        )
        widgets.link((score, "value"), (self, "score"))

        score_description = widgets.Textarea(
            value=self.description,
            description="Description:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_description, "value"), (self, "description"))

        score_idx = widgets.IntText(
            value=self.group_id,
            description="Class ID:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_idx, "value"), (self, "group_id"))

        is_active = widgets.ToggleButton(
            value=self.is_active,
            description="Set Default",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((is_active, "value"), (self, "is_active"))

        def on_active_change(change):
            is_active = change["new"]
            vis = "visible" if change["new"] else "hidden"
            score.layout.visibility = vis
            score_idx.layout.visibility = vis
            score_description.layout.visibility = vis

        is_active.observe(on_active_change, names="value")
        on_active_change({"new": is_active.value})

        categorical_row_config = widgets.GridspecLayout(
            1, 12, layout=widgets.Layout(width="100%", grid_gap="5px")
        )
        categorical_row_config[0, 0] = is_active
        categorical_row_config[0, 1:4] = score
        categorical_row_config[0, 4:7] = score_idx
        categorical_row_config[0, 7:] = score_description

        if should_display:
            display(categorical_row_config)
        return categorical_row_config


class RangeScorePatternViewModel(traitlets.HasTraits):
    lower_is_inf = traitlets.Bool(False)
    upper_is_inf = traitlets.Bool(False)
    lower = traitlets.Float(0)
    upper = traitlets.Float(0)
    group_id = traitlets.Int(0)
    score = traitlets.Float(0)
    description = traitlets.Unicode("")

    @staticmethod
    def from_pydantic_model(model: "RangeScorePattern"):
        lower_is_inf = model.range.start == -float("inf")
        upper_is_inf = model.range.end == float("inf")
        return RangeScorePatternViewModel(
            lower_is_inf=lower_is_inf,
            upper_is_inf=upper_is_inf,
            lower=0 if lower_is_inf else model.range.start,
            upper=0 if upper_is_inf else model.range.end,
            group_id=model.group_id,
            score=model.score,
            description=model.description,
        )

    def to_pydantic_model(self) -> DefaultScorePattern:
        st = -float("inf") if self.lower_is_inf else self.lower
        ed = float("inf") if self.upper_is_inf else self.upper
        return RangeScorePattern(
            range=(st, ed),
            group_id=self.group_id,
            score=self.score,
            description=self.description,
        )

    def get_widget(self, should_display=True, on_delete_callback=None):
        score = widgets.FloatText(
            value=self.score, description="Score:", layout=widgets.Layout(width="auto")
        )
        widgets.link((score, "value"), (self, "score"))

        score_description = widgets.Textarea(
            value=self.description,
            description="Description:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_description, "value"), (self, "description"))

        score_idx = widgets.IntText(
            value=self.group_id,
            description="Class ID:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_idx, "value"), (self, "group_id"))

        delete_row = widgets.Button(
            # description='',
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(
                right="5px",
                top="5px",
                width="32px",
                height="28px",
            ),
        )
        if on_delete_callback is not None:
            delete_row.on_click(lambda evnt: on_delete_callback(self))

        upper_bound = widgets.FloatText(
            value=self.upper, description="", layout=widgets.Layout(width="auto")
        )
        widgets.link((upper_bound, "value"), (self, "upper"))
        lower_bound = widgets.FloatText(
            value=self.lower, description="", layout=widgets.Layout(width="auto")
        )
        widgets.link((lower_bound, "value"), (self, "lower"))

        upper_inf = widgets.ToggleButton(
            value=self.upper_is_inf,
            description=r"inf",
            layout=widgets.Layout(width="4em"),
        )
        widgets.link((upper_inf, "value"), (self, "upper_is_inf"))
        lower_inf = widgets.ToggleButton(
            value=self.lower_is_inf,
            description=r"-inf",
            layout=widgets.Layout(width="4em"),
        )
        widgets.link((lower_inf, "value"), (self, "lower_is_inf"))

        def on_set_bounds(cmp, change):
            is_not_infty = not change["new"]
            vis = "visible" if is_not_infty else "hidden"
            cmp.layout.visibility = vis

        upper_inf.observe(
            lambda change: on_set_bounds(upper_bound, change), names="value"
        )
        on_set_bounds(upper_bound, {"new": upper_inf.value})
        lower_inf.observe(
            lambda change: on_set_bounds(lower_bound, change), names="value"
        )
        on_set_bounds(lower_bound, {"new": lower_inf.value})

        categorical_row_config_grid = widgets.GridspecLayout(
            1, 12, layout=widgets.Layout(width="100%", grid_gap="5px")
        )
        categorical_row_config_grid[0, 0:2] = score
        categorical_row_config_grid[0, 2:4] = score_idx
        categorical_row_config_grid[0, 4:8] = score_description
        categorical_row_config_grid[0, 8:10] = widgets.Box(
            [widgets.Label("Min: "), lower_inf, lower_bound],
            layout=widgets.Layout(
                width="auto", display="grid", grid_template_columns="2.5em auto auto"
            ),
        )
        categorical_row_config_grid[0, 10:] = widgets.Box(
            [
                widgets.Label("Max: ", layout=widgets.Layout(width="4em")),
                upper_inf,
                upper_bound,
            ],
            layout=widgets.Layout(
                width="auto", display="grid", grid_template_columns="2.5em auto auto"
            ),
        )

        delete_row = widgets.Button(
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(
                width="32px",
                height="28px",
            ),
        )
        if on_delete_callback is not None:
            delete_row.on_click(lambda evnt: on_delete_callback(self))
        categorical_row_config = widgets.HBox(
            [categorical_row_config_grid, delete_row],
            layout=widgets.Layout(width="100%", border="1px solid grey"),
        )
        if should_display:
            display(categorical_row_config)
        return categorical_row_config


class ScoreCriteriaCategoricalViewModel(traitlets.HasTraits):
    variable_type = "Categorical"
    variable = traitlets.Unicode()
    default_behavior = traitlets.Unicode()
    discrete_scores = traitlets.List()

    def __init__(self, other_score=None, **traits):
        super().__init__(**traits)
        if not isinstance(other_score, DefaultScorePatternViewModel):
            other_score = DefaultScorePatternViewModel.from_pydantic_model(other_score)
        self.other_score = other_score

    @staticmethod
    def from_pydantic_model(model: "ScoreCriteriaCategorical"):
        return ScoreCriteriaCategoricalViewModel(
            variable=model.variable,
            default_behavior=model.default_behavior,
            discrete_scores=[
                DiscreteScorePatternViewModel.from_pydantic_model(v)
                for v in model.discrete_scores
            ],
            other_score=model.other_score,
        )

    def to_pydantic_model(self) -> ScoreCriteriaCategorical:
        return ScoreCriteriaCategorical(
            variable=self.variable,
            default_behavior=self.default_behavior,
            discrete_scores=[v.to_pydantic_model() for v in self.discrete_scores],
            other_score=self.other_score.to_pydantic_model(),
        )

    def get_widget(self, should_display=True, on_variable_update=None):
        # Header Components
        ## Variable Name
        variable_name = widgets.Text(
            value=self.variable,
            placeholder="Variable to edit",
            description="Variable:",
            layout=widgets.Layout(width="100%"),
        )
        widgets.link((variable_name, "value"), (self, "variable"))
        if on_variable_update is not None:
            variable_name.observe(on_variable_update, "value")

        # def update_accordian_name(change):
        #     nv = change['new']
        #     if nv == '':
        #         accordian.titles = (DEFAULT_TITLE,)
        #     else:
        #         accordian.titles = (f"{nv} (Categorical)",)

        ## Default Behaviour
        default_behaviour = widgets.Dropdown(
            value="matches",
            options=[
                "regex",
                "regex_end",
                "regex_start",
                "regex_partial",
                "matches",
                "matches_end",
                "matches_start",
                "matches_partial",
            ],
            description="Behaviour:",
            ensure_option=True,
            layout=widgets.Layout(width="100%"),
        )
        widgets.link((default_behaviour, "value"), (self, "default_behavior"))

        ## Add Button
        add_btn = widgets.Button(
            icon="plus",
            layout=widgets.Layout(
                width="45px",
                height="28px",
            ),
        )

        def add_dv(evnt):
            new_item = DiscreteScorePatternViewModel()
            new_view = new_item.get_widget(
                should_display=False, on_delete_callback=delete_discrete_var
            )
            # Have to make new list for traitlet to emit event
            self.discrete_scores = [*self.discrete_scores, new_item]
            discrete_box.children = discrete_box.children + (new_view,)

        add_btn.on_click(add_dv)

        header = widgets.HBox(
            [variable_name, default_behaviour, add_btn],
            layout=widgets.Layout(width="100%"),
        )

        # Discrete Vars
        def delete_discrete_var(item):
            new_discrete_scores = []
            children = []
            for c, v in zip(discrete_box.children, self.discrete_scores):
                if id(v) == id(item):
                    continue
                new_discrete_scores.append(v)
                children.append(c)

            self.discrete_scores = new_discrete_scores
            discrete_box.children = tuple(children)

        discrete_box = widgets.VBox(
            [
                v.get_widget(
                    should_display=False, on_delete_callback=delete_discrete_var
                )
                for v in self.discrete_scores
            ],
            layout=widgets.Layout(width="100%"),
        )

        view = widgets.VBox(
            [header, discrete_box, self.other_score.get_widget(should_display=False)],
            layout=widgets.Layout(width="100%"),
        )

        if should_display:
            display(view)
        return view


class ScoreCriteriaNumericalViewModel(traitlets.HasTraits):
    variable_type = "Numerical"
    variable = traitlets.Unicode()
    discrete_scores = traitlets.List()
    range_scores = traitlets.List()
    lower_bound_inclusive = traitlets.Bool(True)
    upper_bound_inclusive = traitlets.Bool(False)

    def __init__(self, other_score=None, **traits):
        super().__init__(**traits)
        if not isinstance(other_score, DefaultScorePatternViewModel):
            other_score = DefaultScorePatternViewModel.from_pydantic_model(other_score)
        self.other_score = other_score

    @staticmethod
    def from_pydantic_model(model: ScoreCriteriaNumerical):
        return ScoreCriteriaNumericalViewModel(
            variable=model.variable,
            lower_bound_inclusive=Bounds.LOWER in model.included_bounds,
            upper_bound_inclusive=Bounds.UPPER in model.included_bounds,
            discrete_scores=[
                DiscreteFloatScorePatternViewModel.from_pydantic_model(v)
                for v in model.discrete_scores
            ],
            range_scores=[
                RangeScorePatternViewModel.from_pydantic_model(v)
                for v in model.range_scores
            ],
            other_score=model.other_score,
        )

    def to_pydantic_model(self) -> ScoreCriteriaNumerical:
        bounds = []
        if self.lower_bound_inclusive:
            bounds.append(Bounds.LOWER)
        if self.upper_bound_inclusive:
            bounds.append(Bounds.UPPER)
        return ScoreCriteriaNumerical(
            variable=self.variable,
            included_bounds=tuple(bounds),
            discrete_scores=[v.to_pydantic_model() for v in self.discrete_scores],
            range_scores=[v.to_pydantic_model() for v in self.range_scores],
            other_score=self.other_score.to_pydantic_model(),
        )

    def get_widget(self, should_display=True, on_variable_update=None):
        # Header Components
        ## Variable Name
        variable_name = widgets.Text(
            value=self.variable,
            placeholder="Variable to edit",
            description="Variable:",
            layout=widgets.Layout(width="100%"),
        )
        widgets.link((variable_name, "value"), (self, "variable"))
        if on_variable_update is not None:
            variable_name.observe(on_variable_update, "value")

        # def update_accordian_name(change):
        #     nv = change['new']
        #     if nv == '':
        #         accordian.titles = (DEFAULT_TITLE,)
        #     else:
        #         accordian.titles = (f"{nv} (Categorical)",)

        ## Default Behaviour
        lower_bound = widgets.ToggleButton(
            value=self.lower_bound_inclusive,
            description="Lower",
            layout=widgets.Layout(width="8em"),
        )
        widgets.link((lower_bound, "value"), (self, "lower_bound_inclusive"))
        upper_bound = widgets.ToggleButton(
            value=self.upper_bound_inclusive,
            description="Upper",
            layout=widgets.Layout(width="8em"),
        )
        widgets.link((upper_bound, "value"), (self, "upper_bound_inclusive"))

        ## Add DV Button
        add_btn = widgets.Button(
            description="Discrete",
            icon="plus",
            layout=widgets.Layout(
                width="12em",
                height="28px",
            ),
        )

        def add_dv(evnt):
            new_item = DiscreteFloatScorePatternViewModel()
            new_view = new_item.get_widget(
                should_display=False, on_delete_callback=delete_discrete_var
            )
            # Have to make new list for traitlet to emit event
            self.discrete_scores = [*self.discrete_scores, new_item]
            discrete_box.children = discrete_box.children + (new_view,)

        add_btn.on_click(add_dv)

        ## Add Range Button
        add_range_btn = widgets.Button(
            description="Range",
            icon="plus",
            layout=widgets.Layout(
                width="12em",
                height="28px",
            ),
        )

        def add_rv(evnt):
            new_item = RangeScorePatternViewModel()
            new_view = new_item.get_widget(
                should_display=False, on_delete_callback=delete_range_var
            )
            # Have to make new list for traitlet to emit event
            self.range_scores = [*self.range_scores, new_item]
            range_box.children = range_box.children + (new_view,)

        add_range_btn.on_click(add_rv)

        header = widgets.HBox(
            [
                variable_name,
                widgets.Label("Included Bounds: ", layout=widgets.Layout(width="12em")),
                upper_bound,
                lower_bound,
                add_range_btn,
                add_btn,
            ],
            layout=widgets.Layout(width="100%"),
        )

        # Discrete Vars
        def delete_discrete_var(item):
            new_discrete_scores = []
            children = []
            for c, v in zip(discrete_box.children, self.discrete_scores):
                if id(v) == id(item):
                    continue
                new_discrete_scores.append(v)
                children.append(c)

            self.discrete_scores = new_discrete_scores
            discrete_box.children = tuple(children)

        discrete_box = widgets.VBox(
            [
                v.get_widget(
                    should_display=False, on_delete_callback=delete_discrete_var
                )
                for v in self.discrete_scores
            ],
            layout=widgets.Layout(width="100%"),
        )

        # RangeVars Vars
        def delete_range_var(item):
            new_scores = []
            children = []
            for c, v in zip(range_box.children, self.range_scores):
                if id(v) == id(item):
                    continue
                new_scores.append(v)
                children.append(c)

            self.range_scores = new_scores
            range_box.children = tuple(children)

        range_box = widgets.VBox(
            [
                v.get_widget(should_display=False, on_delete_callback=delete_range_var)
                for v in self.range_scores
            ],
            layout=widgets.Layout(width="100%"),
        )

        view = widgets.VBox(
            [
                header,
                range_box,
                discrete_box,
                self.other_score.get_widget(should_display=False),
            ],
            layout=widgets.Layout(width="100%"),
        )

        if should_display:
            display(view)
        return view


class ScoreCardViewModel(traitlets.HasTraits):
    bin_prefix = traitlets.Unicode("SCORE_BIN_")
    score_prefix = traitlets.Unicode("SCORE_VALUE_")
    description_prefix = traitlets.Unicode("SCORE_DESC_")
    variable_params = traitlets.List()

    @classmethod
    def from_pydantic_model(cls, model: ScoreCardModel):
        variable_params = [
            (
                ScoreCriteriaCategoricalViewModel.from_pydantic_model(vp)
                if isinstance(vp, ScoreCriteriaCategorical)
                else ScoreCriteriaNumericalViewModel.from_pydantic_model(vp)
            )
            for vp in model.variable_params
        ]
        return cls(
            bin_prefix=model.bin_prefix,
            score_prefix=model.score_prefix,
            description_prefix=model.description_prefix,
            variable_params=variable_params,
        )

    def to_pydantic_model(self) -> ScoreCardModel:
        variable_params = [vpm.to_pydantic_model() for vpm in self.variable_params]
        return ScoreCardModel(
            bin_prefix=self.bin_prefix,
            score_prefix=self.score_prefix,
            description_prefix=self.description_prefix,
            variable_params=variable_params,
        )

    def get_widget(self, should_display=True):
        # Header Section
        header_grid = widgets.GridspecLayout(2, 6)
        bin_prefix_w = widgets.Text(
            value=self.bin_prefix,
            placeholder="Value used as prefix for output bins",
            description="Bin Prefix:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((bin_prefix_w, "value"), (self, "bin_prefix"))
        score_prefix_w = widgets.Text(
            value=self.score_prefix,
            placeholder="Value used as prefix for output bins",
            description="Value Prefix:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((score_prefix_w, "value"), (self, "score_prefix"))
        description_prefix_w = widgets.Text(
            value=self.description_prefix,
            placeholder="Value used as prefix for output bins",
            description="Score Prefix:",
            layout=widgets.Layout(width="auto"),
        )
        widgets.link((description_prefix_w, "value"), (self, "description_prefix"))
        header_grid[0, 0:2] = bin_prefix_w
        header_grid[0, 2:4] = score_prefix_w
        header_grid[0, 4:] = description_prefix_w
        add_categorical_button = widgets.Button(
            description="Add Categorical", layout=widgets.Layout(width="auto")
        )
        add_numerical_button = widgets.Button(
            description="Add Numerical", layout=widgets.Layout(width="auto")
        )
        header_grid[1, 0:3] = add_categorical_button
        header_grid[1, 3:] = add_numerical_button

        # Criteria Accordians
        def update_accordian_name(accordian, nv, variable_type):
            if nv == "":
                accordian.titles = (f"{variable_type} Variable",)
            else:
                accordian.titles = (f"{nv} ({variable_type})",)

        def delete_variable_param(id_param):
            new_vp = []
            new_children = []
            # Maybe not the most efficient approach with with less than 100 odd rules this should be sufficient
            for el, vp in zip(
                variable_parameter_container.children, self.variable_params
            ):
                if id(vp) == id_param:
                    continue
                new_vp.append(vp)
                new_children.append(el)
            variable_parameter_container.children = new_children
            self.variable_params = new_vp

        def get_vp_widget(vp):
            # Create Wrapper accordian
            accordion = widgets.Accordion(layout=widgets.Layout(width="100%"))
            variable_type = vp.variable_type
            child_widget = vp.get_widget(
                should_display=False,
                on_variable_update=lambda change: update_accordian_name(
                    accordion, change["new"], variable_type
                ),
            )
            accordion.children = (child_widget,)
            update_accordian_name(accordion, vp.variable, variable_type)
            # Add delete functionality
            del_btn = widgets.Button(
                button_style="danger",
                icon="trash",
                layout=widgets.Layout(
                    right="5px",
                    top="5px",
                    width="32px",
                    height="28px",
                ),
            )
            del_btn.on_click(lambda evnt: delete_variable_param(id(vp)))

            return widgets.Box([accordion, del_btn])

        variable_parameter_children = []
        for vp in self.variable_params:
            variable_parameter_children.append(get_vp_widget(vp))

        variable_parameter_container = widgets.VBox(variable_parameter_children)

        # Hookup add functionality
        def add_categorical(evnt):
            model = ScoreCriteriaCategoricalViewModel()
            view = get_vp_widget(vp)
            self.variable_params.append(model)
            variable_parameter_container.children.append(view)

        add_categorical_button.on_click(add_categorical)

        def add_numerical():
            model = ScoreCriteriaNumericalViewModel()
            view = get_vp_widget(vp)
            self.variable_params.append(model)
            variable_parameter_container.children.append(view)

        add_numerical_button.on_click(add_numerical)

        view = widgets.VBox(
            [header_grid, variable_parameter_container],
            layout=widgets.Layout(width="100%"),
        )
        if should_display:
            display(view)
        return view

    def save(self, file: str):
        self.to_pydantic_model().save(file=file)
