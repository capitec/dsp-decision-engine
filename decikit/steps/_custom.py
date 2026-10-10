from __future__ import annotations

import inspect
import typing as t
from abc import abstractmethod

from decikit.steps._configurable import ConfigurableStep
from decikit.steps._values import ParamRef


def _is_value(annotation: t.Any) -> bool:
    return ParamRef in t.get_args(annotation)


class CustomFunction(ConfigurableStep):
    """A step type whose fields are data and whose logic is one pure function, `impl`.

    Declare `impl` as a `staticmethod` (no `self`, so it compiles like any
    function step) and one field per `impl` argument, named like it. A `str`
    field is the column to read; a `Value[T]` field is a value, which a param
    can also set. The step's `name` is the column `impl` writes. For several
    outputs, annotate `impl` with a `tuple[...]` return and set `outputs`.

    Unlike a function that reads a module-level object or a closure, all of
    the step's data is in its fields, so it dumps, loads, diffs and compiles.
    For a table of rows use `param_table`; for anything `impl` can't express,
    write `to_ir` yourself on a `ConfigurableStep`.

    Example::

        from typing import Literal
        from decikit import CustomFunction, Value

        class Cap(CustomFunction):
            type: Literal["cap"] = "cap"
            x: str
            limit: Value[float] = 100.0

            @staticmethod
            def impl(x: float, limit: float) -> float:
                return min(x, limit)

        Cap(name="capped", x="amount", limit=50.0).run(df)       # writes column "capped"
    """

    outputs: list[str] | None = None

    @staticmethod
    @abstractmethod
    def impl(*args: t.Any) -> t.Any: ...

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: t.Any) -> None:
        raw = inspect.getattr_static(cls, "impl")
        if not isinstance(raw, staticmethod) or raw.__isabstractmethod__:
            raise TypeError(f"{cls.__name__} must define `impl` as a @staticmethod, the pure function the step "
                            "calls; it has no `self`, so numba can compile it.")
        for name in inspect.signature(cls.impl).parameters:
            field = cls.model_fields.get(name)
            if field is None:
                raise TypeError(f"{cls.__name__}.impl has an argument {name!r} but the class has no field "
                                f"{name!r}; add `{name}: str` (the column to read) or `{name}: Value[float]`.")
            if field.annotation is not str and not _is_value(field.annotation):
                raise TypeError(f"{cls.__name__}.{name} is {field.annotation}; a CustomFunction field must be `str` "
                                "(a column to read) or `Value[T]` (a value). For a table use `param_table`; for "
                                "other data write `to_ir` on a ConfigurableStep, or load it from a config document.")
        super().__pydantic_init_subclass__(**kwargs)

    def to_ir(self, ctx: t.Any) -> t.Any:
        inputs, values = {}, {}
        for name in inspect.signature(self.impl).parameters:
            (values if _is_value(type(self).model_fields[name].annotation) else inputs)[name] = getattr(self, name)
        return ctx.call(self, self.impl, inputs=inputs, values=values, outputs=self.outputs)
