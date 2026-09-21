import typing as t
import inspect
import polars as pl
from pydantic import BaseModel
from types import ModuleType
from decider.types import TInputType, TOutputType
from .core import BaseModule
from .expression import ExpressionModule, Node
from .primitives.join import FrameModule

if t.TYPE_CHECKING:
    from decider.executor import Executor


_FRAME_ANNOTATIONS = frozenset({pl.DataFrame, pl.LazyFrame})


def _detect_module_kind(functions: t.Tuple[t.Callable, ...]) -> t.Literal["expr", "frame"]:
    """Inspect all non-config parameters; assert they are consistently pl.Expr or pl.DataFrame."""
    kind: t.Optional[str] = None
    for func in functions:
        for param_name, param in inspect.signature(func).parameters.items():
            if param_name == "config":
                continue
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param_name}' in '{func.__name__}' has no annotation. "
                    "All non-config parameters must be annotated as pl.Expr or pl.DataFrame/pl.LazyFrame."
                )
            if ann is pl.Expr:
                param_kind = "expr"
            elif ann in _FRAME_ANNOTATIONS:
                param_kind = "frame"
            else:
                raise TypeError(
                    f"Parameter '{param_name}' in '{func.__name__}' is annotated as {ann!r}. "
                    "All non-config parameters must be pl.Expr or pl.DataFrame/pl.LazyFrame."
                )
            if kind is None:
                kind = param_kind
            elif kind != param_kind:
                raise TypeError(
                    "Mixed parameter types: all non-config parameters must consistently be "
                    "pl.Expr or pl.DataFrame/pl.LazyFrame across all functions."
                )
    return kind or "expr"


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
    kind = _detect_module_kind(functions)

    # Collect config class and per-function injection flags.
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

    if config_class is not None and not issubclass(config_class, BaseModel):
        raise TypeError(f"The 'config' parameter must be a subclass of pydantic.BaseModel. Found {config_class}")

    if config_class is None:
        config_class = BaseModel

    module_name = module_name.lower()

    if kind == "expr":
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
                    for k in list(node.input_map.keys()):
                        if k in internal_nodes:
                            node.input_map[k] = internal_nodes[k]

                return internal_nodes

        return TModule

    # Pre-compute param names (excluding "config") once at class-generation time.
    frame_steps: t.List[t.Tuple[t.Callable, t.List[str], bool]] = [
        (
            func,
            [p for p in inspect.signature(func).parameters if p != "config"],
            inject,
        )
        for func, inject in zip(functions, requires_injection)
    ]

    # kind == "frame"
    class TFrameModule(FrameModule, config_class):
        type: t.Literal[module_name]

        def execute(self, inputs: TInputType, _executor: "Executor") -> TOutputType:
            nonlocal frame_steps
            # Results from earlier functions become available as named frames for later ones.
            frames: t.Dict[str, t.Any] = dict(inputs)
            result = None
            for func, param_names, inject in frame_steps:
                kwargs: t.Dict[str, t.Any] = {"config": self} if inject else {}
                for param_name in param_names:
                    if param_name not in frames:
                        raise ValueError(
                            f"Frame '{param_name}' required by '{func.__name__}' not found. "
                            f"Available: {list(frames.keys())}"
                        )
                    val = frames[param_name]
                    kwargs[param_name] = val.lazy() if isinstance(val, pl.DataFrame) else val
                result = func(**kwargs)
                frames[func.__name__] = result
            return result

    return TFrameModule

def generate_from_module(module_name: str, module: ModuleType) -> t.Type[BaseModule]:
    """Dynamically generate a BaseModule subclass from all the functions in a given module."""
    functions = [getattr(module, attr) for attr in dir(module) if callable(getattr(module, attr))]
    return generate_from_functions(module_name, *functions)