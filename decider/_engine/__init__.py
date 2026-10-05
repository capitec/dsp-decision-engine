from decider._engine.ir.context import IRContext, step_map, to_ir

__all__ = ["Engine", "IRContext", "step_map", "to_ir"]


def __getattr__(name: str):
    # Engine pulls in numba; `import decider` must not.
    if name == "Engine":
        from decider._engine.run import Engine

        return Engine
    raise AttributeError(name)
