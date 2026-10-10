"""`Step.run` and `compile` follow a context-scoped default engine, fused when unset; `session` and serving do not."""
import asyncio
import contextvars
import sys
import threading
import warnings

import polars as pl
import pytest

import decikit
from decikit import FusedEngine, InterpretedEngine, SteppedEngine, flow
from decikit._engine.run.engines import default_engine
from decikit.config import JsonFileStore
from decikit.exceptions import EngineError, FallbackWarning
from decikit.serving import RequestHandler


def label(code: str) -> bool:  # reads a str, which no kernel takes: it runs in Python
    return code.startswith("A")


pipeline = flow(label, name="labels")
FRAME = pl.DataFrame({"code": ["AB", "CD"]})


def _fallback_warnings(run) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
    return [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]


def test_the_default_is_fused_when_nothing_sets_it():
    assert contextvars.Context().run(default_engine) == FusedEngine()
    assert contextvars.Context().run(pipeline.compile).engine == FusedEngine()


def test_the_block_sets_the_default_and_resets_it_on_exit():
    with decikit.engine("interpreted"):
        assert default_engine() == InterpretedEngine()
        assert pipeline.compile().engine == InterpretedEngine()
    assert default_engine() == FusedEngine()


def test_the_block_resets_the_default_when_it_raises():
    with pytest.raises(ZeroDivisionError):
        with decikit.engine("interpreted"):
            1 / 0
    assert default_engine() == FusedEngine()


def test_nested_blocks_restore_the_outer_engine():
    with decikit.engine("interpreted"):
        with decikit.engine("stepped"):
            assert default_engine() == SteppedEngine()
        assert default_engine() == InterpretedEngine()
    assert default_engine() == FusedEngine()


def test_an_explicit_engine_wins_over_the_block():
    with decikit.engine("interpreted"):
        assert pipeline.compile("stepped").engine == SteppedEngine()
        assert _fallback_warnings(lambda: pipeline.run(FRAME, engine="fused")) != []


def test_run_follows_the_default_engine_instance():
    with decikit.engine(FusedEngine(strict_compile=True)):
        with pytest.raises(EngineError, match="labels/label runs in Python"):
            pipeline.run(FRAME)
    with decikit.engine("interpreted"):
        assert _fallback_warnings(lambda: pipeline.run(FRAME)) == []   # interpreted runs Python by design


def test_a_str_step_warns_once_under_the_default():
    assert len(_fallback_warnings(lambda: pipeline.run(FRAME))) == 1
    exe = pipeline.compile()
    assert len(_fallback_warnings(lambda: (exe.run(FRAME), exe.run(FRAME)))) == 1


def test_a_lazy_frame_runs_on_polars_whatever_the_default_and_a_named_engine_collects_it():
    strict = FusedEngine(strict_compile=True)
    with decikit.engine(strict):
        assert isinstance(pipeline.run(FRAME.lazy()), pl.LazyFrame)
        with pytest.raises(EngineError, match="labels/label runs in Python"):
            pipeline.run(FRAME.lazy(), engine=strict)
    out = pipeline.run(FRAME.lazy(), engine="interpreted")
    assert isinstance(out, pl.DataFrame) and out["label"].to_list() == [True, False]


def _in_thread(target) -> list:
    seen = []
    thread = threading.Thread(target=lambda: seen.append(target()))
    thread.start()
    thread.join()
    return seen


def test_a_thread_sees_the_block_only_through_copy_context():
    with decikit.engine("interpreted"):
        context = contextvars.copy_context()
        assert _in_thread(lambda: context.run(default_engine)) == [InterpretedEngine()]
        # A plain thread starts with an empty context, so it sees fused, unless this Python build makes
        # threads inherit their starter's context (`sys.flags.thread_inherit_context`, 3.14 free-threaded).
        plain = InterpretedEngine() if getattr(sys.flags, "thread_inherit_context", 0) else FusedEngine()
        assert _in_thread(default_engine) == [plain]


def test_a_thread_started_before_the_block_keeps_the_default():
    entered, seen = threading.Event(), []
    context = contextvars.copy_context()

    def read():
        entered.wait()
        seen.append(default_engine())

    thread = threading.Thread(target=context.run, args=(read,))
    thread.start()
    with decikit.engine("interpreted"):
        entered.set()
        thread.join()
    assert seen == [FusedEngine()]


def test_an_async_task_inherits_a_copy_of_the_default():
    async def main():
        with decikit.engine("interpreted"):
            inside, resume = asyncio.Event(), asyncio.Event()

            async def child():
                seen = default_engine()
                with decikit.engine("stepped"):
                    inside.set()
                    await resume.wait()
                return seen

            task = asyncio.create_task(child())
            await inside.wait()
            parent = default_engine()   # the child's block changed only its own copy
            resume.set()
            return await task, parent

    assert asyncio.run(main()) == (InterpretedEngine(), InterpretedEngine())


def test_a_session_opens_interpreted_whatever_the_default():
    assert pipeline.session(FRAME).executable.engine == InterpretedEngine()
    with decikit.engine("fused"):
        assert pipeline.session(FRAME).executable.engine == InterpretedEngine()
        assert pipeline.session(FRAME, engine="stepped").executable.engine == SteppedEngine()


def build():
    return pipeline


def test_a_handler_runs_its_own_engine_whatever_the_default(tmp_path):
    JsonFileStore(basepath=str(tmp_path)).create_version({"params": {}})
    with decikit.engine("interpreted"):
        handler = RequestHandler(JsonFileStore(basepath=str(tmp_path)), build)
        handler.stage()
        handler.activate()
        assert handler.engine == FusedEngine()
        assert handler.module_fn().executable.engine == FusedEngine()
        assert handler.score({"code": "AB"})["label"] is True
