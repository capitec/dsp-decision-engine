import typing as t
from pydantic import BaseModel, PrivateAttr
# from polars import pl



class DefinedFunction(BaseModel):
    module_name: str
    function_name: str

    _function: t.Optional[t.Callable[..., t.Any]] = PrivateAttr(default=None)

    @classmethod
    def from_function(cls, fn: t.Callable[..., t.Any]) -> "DefinedFunction":
        return cls(module_name=fn.__module__, function_name=fn.__name__)
    
    def get_function(self) -> t.Callable[..., t.Any]:
        if self._function is None:
            from importlib import import_module
            module = import_module(self.module_name)
            self._function = getattr(module, self.function_name)
        return self._function
    
    def __call__(self, *args, **kwds):
        return self.get_function()(*args, **kwds)

# class SerializableExpression(BaseModel):
#     expression: str

#     def to_polars_expr(self, **inputs) -> pl.Expr:
#         # In a real implementation, you would need to parse the expression string
#         # and convert it into a Polars expression. This is a placeholder.
#         return pl.col(self.expression)
    
#     @classmethod
#     def from_polars_expr_fn(cls, fn: t.Callable[..., pl.Expr]) -> "SerializableExpression":
#         # In a real implementation, you would need to convert a Polars expression
#         # back into a string representation. This is a placeholder.
#         return cls(expression=str(fn))