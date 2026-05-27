import typing as t
import inspect
import polars as pl
from pydantic import BaseModel
from types import ModuleType
from .core import BaseModule
from .expression import ExpressionModule, Node


def generate_from_functions(module_name: str, *functions: t.Callable) -> t.Type[BaseModule]:
    """Create a module class from plain Python functions.

    Three conventions wire everything together automatically:

    1. **Function name → output column name**
       Each function produces a new column whose name matches the function's
       ``__name__``.  ``def risk_score(...) → pl.Expr`` adds a ``risk_score``
       column to the frame.

    2. **Parameter name → input column lookup**
       Parameters are resolved in order:
       a. If another function in this call has the same name, its output
          expression is injected (dependency wiring).
       b. Otherwise the parameter is read from the column of that name in the
          input dataframe.

       Example — ``amount_centered`` receives the output of ``amount_mean``:
       ::

           def amount_mean(amount: pl.Expr) -> pl.Expr:
               return amount.mean()

           def amount_centered(amount: pl.Expr, amount_mean: pl.Expr) -> pl.Expr:
               return amount - amount_mean

    3. **Optional ``config`` parameter → module config injection**
       If a function declares a ``config`` parameter annotated with a Pydantic
       model, the module itself acts as that config (fields are defined on the
       generated class) and the current instance is injected at call time.

    Args:
        module_name: Type discriminator string used for serialisation.  Also
            shows up in debug output and error messages.  Must be lowercase and
            unique within your project (e.g. ``"income_scorer"``).
        *functions: One or more plain functions returning ``pl.Expr``.  Order
            only matters when two functions share the same dependency graph
            level — prefer topological clarity over positional ordering.

    Returns:
        A new ``BaseModule`` subclass.  Instantiate it with ``name=`` to get a
        module you can call directly or compose with ``|`` and ``&``::

            Scorer = generate_from_functions("scorer", risk_score, tier_flag)
            scorer = Scorer(name="my_scorer")
            result = scorer({"input": df})

    Input frame convention:
        Pass frames as a dict.  The key ``"input"`` is the default frame every
        expression targets.  You may pass additional named frames; expression
        functions always operate on columns, not whole frames.

    Quickstart::

        import polars as pl
        from decider.modules.functional import generate_from_functions

        def score(amount: pl.Expr) -> pl.Expr:
            return amount * 100

        Scorer = generate_from_functions("scorer", score)
        result = Scorer(name="s")({"input": pl.DataFrame({"amount": [1, 2, 3]})})
        # result is a LazyFrame; call .collect() to materialise it
    """
    # Create a dictionary of the functions to be used as the namespace for the new class

    # 1. look in all functions for a keyword argument named config and ensure they are all the same
    config_class = None
    requires_injection: t.List[bool] = []
    for func in functions:
        sig = inspect.signature(func)
        if "config" not in sig.parameters:
            requires_injection.append(False)
            continue
        requires_injection.append(True)
        param = sig.parameters["config"]
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Function {func.__name__} has a 'config' parameter but it is not annotated with a type")
        if config_class is None:
            config_class = param.annotation
        elif config_class != param.annotation:
            raise TypeError(f"All functions must have the same type for the 'config' parameter. Found {config_class} and {param.annotation}")
    
    # check config is a pydantic model
    if config_class is not None:
        if not issubclass(config_class, BaseModel):
            raise TypeError(f"The 'config' parameter must be a subclass of pydantic.BaseModel. Found {config_class}")
    
    if config_class is None: config_class = BaseModel

    module_name = module_name.lower()

    class TModule(ExpressionModule, config_class):
        type: t.Literal[module_name]

        def expand_nodes(self) -> t.Dict[str, "Node"]:
            nonlocal functions, requires_injection
            internal_nodes: t.Dict[str, Node] = {}
            for inject, func in zip(requires_injection, functions):
                node = Node.from_callable(
                    func,
                    static_kwargs={"config": self} if inject else None
                )
                internal_nodes[node.name] = node

            for node in internal_nodes.values():
                for k in node.input_map.keys():
                    if k in internal_nodes:
                        node.input_map[k] = internal_nodes[k]

            return internal_nodes
    return TModule

def generate_from_module(module_name: str, module: ModuleType) -> t.Type[BaseModule]:
    """Dynamically generate a BaseModule subclass from all the functions in a given module."""
    functions = [getattr(module, attr) for attr in dir(module) if callable(getattr(module, attr))]
    return generate_from_functions(module_name, *functions)