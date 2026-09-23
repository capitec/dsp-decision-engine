from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from numba.core import event


@contextmanager
def no_recompile() -> Iterator[None]:
    """Fail if numba compiles anything inside the block, e.g. across a retune.

    Only real compilations count: calls served by an existing specialisation
    (or loaded from numba's disk cache) don't.

    Example::

        exe = Engine().bind(pipeline, mode="fused")
        exe.run(df)
        with no_recompile():
            exe.run(df, params={"term": {"cap": 24.0}})
    """
    with event.install_recorder("numba:compile") as recorder:
        yield
    compiled = [f"{e.data['dispatcher'].py_func.__qualname__}{e.data['args']}"
                for _, e in recorder.buffer if e.status is event.EventStatus.START]
    if compiled:
        raise AssertionError(f"numba compiled {len(compiled)} specialisation(s): {', '.join(compiled)}")
