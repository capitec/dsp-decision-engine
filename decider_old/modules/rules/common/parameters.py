import typing as t
import polars as pl
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from ....serializable.schema import PrimitiveSchema


class ParameterInfo(BaseModel):
    """Declares the type and optional default for a single flat-rule parameter.

    Parameters are named values injected into rules at runtime via ``InputRef``.
    They allow thresholds, lookup values, and scalar config to be externalised
    from the rule structure so they can be changed without rebuilding the rule.

    ``type`` accepts any ``PrimitiveSchema`` — see ``PrimitiveSchema`` for the
    full list of supported type expressions.  Quick reference:

    **Scalar types** (plain string)::

        ParameterInfo(type=PrimitiveSchema.model_validate("Float64"), default_value=8.0)
        ParameterInfo(type=PrimitiveSchema.model_validate("String"),  default_value="N")
        ParameterInfo(type=PrimitiveSchema.model_validate("Int64"),   default_value=0)
        ParameterInfo(type=PrimitiveSchema.model_validate("Boolean"), default_value=True)

    **Compound types** (dict with ``"type"`` key)::

        # List of strings — e.g. keyword allow-list
        ParameterInfo(
            type=PrimitiveSchema.model_validate({"type": "List", "inner": "String"}),
            default_value=None,   # required at runtime; no static default
        )

        # List of floats — e.g. score band boundaries
        ParameterInfo(
            type=PrimitiveSchema.model_validate({"type": "List", "inner": "Float64"}),
            default_value=[0.0, 0.5, 1.0],
        )

    The ``default_value`` is optional.  When supplied it must be compatible with
    the declared type; an ``InvalidOperationError`` is raised at construction
    time if it is not.  When ``None``, the parameter must be provided at runtime
    via the ``parameters`` column in the input frame.

    **Using a parameter in a rule** — reference it with ``InputRef``::

        from decider.modules.rules import InputRef, UnaryGreaterThanEqual
        UnaryGreaterThanEqual(op=">=", feature=Feature("age"),
                              threshold=InputRef(key="min_age_threshold"))
    """

    type: PrimitiveSchema
    default_value: t.Optional[t.Any] = None
    _polars_literal: t.Optional[pl.Expr] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            polars_type = self.type.polars_type
            try:
                self._polars_literal = pl.lit(self.default_value, polars_type)
            except pl.exceptions.InvalidOperationError:
                raise ValueError(
                    f"Default value {self.default_value} is not compatible with type {self.type}"
                )

        return self

    @property
    def polars_literal(self) -> t.Optional[pl.Expr]:
        return self._polars_literal


class WithParameters:
    """Mixin that adds named runtime parameters to a flat-rule module.

    Declare parameters on a ``PrioritizedFlatRuleModule`` or ``FlatRuleModule``
    by populating the ``parameters`` dict.  Each key names a parameter;
    the value is a ``ParameterInfo`` that declares its type and optional default.

    At execution time the framework looks for a struct column named
    ``parameters_col`` (default ``"parameters"``) in the input frame.  Each
    field of that struct is matched by name to the declared parameters and
    injected into rules that reference it via ``InputRef``.

    **Example — threshold and keyword list as parameters**::

        from decider.modules.rules import PrioritizedFlatRuleModule, InputRef
        from decider.modules.rules.common.parameters import ParameterInfo
        from decider.serializable.schema import PrimitiveSchema

        m = PrioritizedFlatRuleModule(
            name="my_rule",
            output=output,
            parameters={
                # Scalar float threshold
                "min_score": ParameterInfo(
                    type=PrimitiveSchema.model_validate("Float64"),
                    default_value=0.5,
                ),
                # List of strings — allow-list of category codes
                "allowed_categories": ParameterInfo(
                    type=PrimitiveSchema.model_validate({"type": "List", "inner": "String"}),
                    default_value=None,  # required at runtime
                ),
            },
            rules=[...],
        )

    **Providing parameters at runtime** — add a ``"parameters"`` struct column
    to the input frame::

        import polars as pl
        df = pl.DataFrame({
            "score":    [0.7, 0.3],
            "category": ["A", "B"],
            "parameters": [
                {"min_score": 0.5, "allowed_categories": ["A", "C"]},
                {"min_score": 0.5, "allowed_categories": ["A", "C"]},
            ],
        })
        result = m({"input": df.lazy()})

    If ``default_value`` is set on a ``ParameterInfo``, the parameter is
    optional at runtime — the default is used when the ``parameters`` column
    is absent or the field is null.  Parameters without a default are required.
    """

    parameters_col: str = "parameters"
    parameters: t.Dict[str, ParameterInfo] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self):
        required_parameters = self.get_required_parameters()
        known_parameters = set(self.parameters.keys())
        if not required_parameters.issubset(known_parameters):
            missing = required_parameters - known_parameters
            raise ValueError(f"Missing parameter definitions for: {missing}")
        return self

    @property
    def parameter_schema(self) -> pl.Schema:
        return pl.Schema(
            {name: info.type.polars_type for name, info in self.parameters.items()}
        )
