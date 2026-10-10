from decikit._engine.ir.context import IRContext, step_map, to_ir

__all__ = ["Engine", "FusedEngine", "IRContext", "InterpretedEngine", "NumbaEngine", "SteppedEngine", "step_map",
           "to_ir"]


def __getattr__(name: str):
    # The engines pull in numba; `import decikit` must not.
    if name in ("Engine", "NumbaEngine", "InterpretedEngine", "SteppedEngine", "FusedEngine"):
        from decikit._engine import run

        return getattr(run, name)
    raise AttributeError(name)
