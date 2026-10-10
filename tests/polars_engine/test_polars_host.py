import datetime as dt
import threading

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from decikit import FusedEngine, PolarsEngine, branch, flow, frame_step, param, step
from decikit.exceptions import EngineError, ParamsError


def ratio(income: float, debt: float) -> float:
    return debt / income


def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


@frame_step(reads=["income"], writes=["rank"])
def rank_income(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("income").rank().alias("rank"))


def flag(rank: float, affordable: bool, min_rank: float = param(2.0)) -> bool:
    return affordable and rank > min_rank


PIPE = flow(ratio, affordable, rank_income, flag, name="p")
DF = pl.DataFrame({"income": [1000.0 * (i + 1) for i in range(100)], "debt": [200.0 + i for i in range(100)]})


def fused(pipe, df=DF, params=None):
    return FusedEngine().bind(pipe).run(df, params)


def streamed(lf: pl.LazyFrame) -> pl.DataFrame:
    pl.Config.set_streaming_chunk_size(7)
    return lf.collect(engine="streaming")


def test_pipeline_without_frame_steps_is_one_streamable_segment():
    exe = PolarsEngine().bind(flow(ratio, affordable, name="p"))
    assert exe.segments == ((("p/ratio", "p/affordable"), True),)
    assert_frame_equal(streamed(exe.run(DF.lazy())), fused(flow(ratio, affordable, name="p")))


def test_frame_step_runs_once_on_the_whole_frame_while_segments_run_per_batch(monkeypatch):
    exe = PolarsEngine().bind(PIPE)
    assert exe.segments == ((("p/ratio", "p/affordable"), True), (("p/rank_income",), False), (("p/flag",), True))
    calls = [0, 0, 0]
    for i, seg in enumerate(exe._segments):
        def counted(df, params=None, trace=None, i=i, run=seg.executable.run):
            calls[i] += df.height > 0
            return run(df, params, trace)
        monkeypatch.setattr(seg.executable, "run", counted)
    out = streamed(exe.run(DF.lazy()))
    assert calls[0] > 1 and calls[1] == 1 and calls[2] > 1
    calls[:] = [0, 0, 0]
    assert_frame_equal(exe.run(DF.lazy()).collect(), out)
    assert calls == [1, 1, 1]


def test_streaming_collect_and_in_memory_collect_give_the_same_frame():
    lf = PolarsEngine().bind(PIPE).run(DF.lazy())
    assert_frame_equal(streamed(lf), lf.collect())
    assert_frame_equal(lf.collect(), fused(PIPE))


def test_sink_parquet_round_trip_equals_fused(tmp_path):
    pl.Config.set_streaming_chunk_size(7)
    DF.write_parquet(tmp_path / "in.parquet")
    PolarsEngine().bind(PIPE).run(pl.scan_parquet(tmp_path / "in.parquet")).sink_parquet(tmp_path / "out.parquet")
    assert_frame_equal(pl.read_parquet(tmp_path / "out.parquet"), fused(PIPE))


def test_a_filter_or_slice_on_the_plan_never_moves_below_a_frame_step(monkeypatch):
    # polars before 1.3x let predicates, projections and slices pass a map_batches node unless told not to.
    passing = pl.LazyFrame.map_batches

    def old_defaults(self, function, **kwargs):
        for k in ("predicate_pushdown", "projection_pushdown", "slice_pushdown"):
            kwargs.setdefault(k, True)
        return passing(self, function, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "map_batches", old_defaults)
    lf, want = PolarsEngine().bind(PIPE).run(DF.lazy()), fused(PIPE)
    assert_frame_equal(lf.filter(pl.col("income") > 50_000).collect(), want.filter(pl.col("income") > 50_000))
    assert_frame_equal(lf.slice(40, 20).collect(), want.slice(40, 20))
    assert_frame_equal(lf.select("flag").collect(), want.select("flag"))


def test_intermediate_columns_are_dropped_as_fused_drops_them():
    lf = PolarsEngine().bind(PIPE).run(DF.lazy())
    assert lf.collect_schema().names() == ["income", "debt", "flag"] == fused(PIPE).columns


def test_params_document_is_pruned_per_segment():
    exe = PolarsEngine().bind(PIPE)
    doc = {"p": {"affordable": {"limit": 0.3}, "flag": {"min_rank": 50.0}}}
    assert_frame_equal(streamed(exe.run(DF.lazy(), doc)), fused(PIPE, params=doc))
    with pytest.raises(ParamsError) as numba:
        fused(PIPE, params={"p": {"nope": {"x": 1}}})
    with pytest.raises(ParamsError) as host:
        exe.run(DF.lazy(), {"p": {"nope": {"x": 1}}})   # before anything is collected
    assert str(host.value) == str(numba.value)


def test_shared_param_used_by_one_segment_only():
    def flag_shared(rank: float, affordable: bool, min_rank: float = param(2.0, shared_key="min_rank")) -> bool:
        return affordable and rank > min_rank

    pipe = flow(ratio, affordable, rank_income, flag_shared, name="p")
    doc = {"shared": {"min_rank": 60.0}, "p": {"affordable": {"limit": 0.3}}}
    assert_frame_equal(streamed(PolarsEngine().bind(pipe).run(DF.lazy(), doc)), fused(pipe, params=doc))


def score(income: float) -> float:
    return income * 2


def score_read(score: float) -> float:
    return score + 1


def score_blind(debt: float) -> float:
    return debt + 1


def test_name_read_and_rewritten_after_a_frame_step():
    pipe = flow(score, rank_income, step(score_read).relabel(writes={"score_read": "score"}), name="q")
    assert_frame_equal(streamed(PolarsEngine().bind(pipe).run(DF.lazy())), fused(pipe))


def test_name_overwritten_after_a_frame_step_is_dropped_first():
    pipe = flow(score, rank_income, step(score_blind).relabel(writes={"score_blind": "score"}), name="q")
    assert_frame_equal(streamed(PolarsEngine().bind(pipe).run(DF.lazy())), fused(pipe))


def test_value_read_then_rewritten_by_a_later_step_after_a_frame_step_splits_the_segment():
    def logged(score: float) -> float:
        return score + 0.5

    pipe = flow(score, rank_income, logged, step(score_read).relabel(writes={"score_read": "score"}), name="q")
    exe = PolarsEngine().bind(pipe)
    assert exe.segments == ((("q/score",), True), (("q/rank_income",), False), (("q/logged",), True),
                            (("q/score_read",), True))
    assert_frame_equal(streamed(exe.run(DF.lazy())), fused(pipe))


def test_value_read_then_rewritten_inside_one_flow_after_a_frame_step_is_refused():
    def logged(score: float) -> float:
        return score + 0.5

    inner = flow(logged, step(score_read).relabel(writes={"score_read": "score"}), name="inner")
    with pytest.raises(EngineError, match="runs q/inner as one segment, which does not wire on its own"):
        PolarsEngine().bind(flow(score, rank_income, inner, name="q"))


def test_name_like_one_a_later_segment_writes_is_no_typo():
    def score_b(income: float) -> float:
        return income

    def score_a(rank: float) -> float:
        return rank * 3

    def total(score_b: float, score_a: float) -> float:
        return score_b + score_a

    pipe = flow(score_b, rank_income, score_a, total, name="t")
    assert_frame_equal(streamed(PolarsEngine().bind(pipe).run(DF.lazy())), fused(pipe))


def test_frame_step_inside_a_branch_makes_the_whole_branch_a_whole_frame_segment():
    def small(income: float) -> bool:
        return income <= 50_000

    def low_rank(income: float) -> float:
        return 0.0

    # The merged `rank` takes the first arm's type: float, where the frame step's is unknown.
    ranked = branch(small, step(low_rank).relabel(writes={"low_rank": "rank"}), rank_income, modifies=["rank"], name="b")
    pipe = flow(ratio, affordable, ranked, flag, name="p")
    exe = PolarsEngine().bind(pipe)
    assert [s for _, s in exe.segments] == [True, False, True]
    assert_frame_equal(streamed(exe.run(DF.lazy())), fused(pipe))


def test_names_written_inside_a_branch_arm_stay_inside_it():
    def big(income: float) -> bool:
        return income > 50_000

    def half(income: float) -> float:
        return income / 2

    def capped(half: float) -> float:
        return min(half, 30_000.0)

    pipe = flow(branch(big, flow(half, capped, name="cap"), modifies=["capped"], name="b"), ratio, affordable,
                rank_income, flag, name="p")
    assert_frame_equal(streamed(PolarsEngine().bind(pipe).run(DF.lazy())), fused(pipe))


def test_frame_step_that_cannot_run_on_zero_rows_gets_a_clear_error():
    @frame_step(reads=["income"], writes=["top"])
    def top_income(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(top=pl.lit(df["income"][0]))

    with pytest.raises(EngineError, match="frame step p/top_income raised on an empty frame.*accept an empty frame"):
        PolarsEngine().bind(flow(top_income, name="p")).run(DF.lazy())


def test_frame_step_column_empty_on_zero_rows_takes_its_declared_dtype():
    def labelled(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.Series("label", ["big" if x > 95_000 else None for x in df["income"]]))

    undeclared = flow(frame_step(labelled, reads=["income"], writes=["label"]), name="p")
    with pytest.raises(EngineError, match="'label', written by p/labelled.*writes=\\{'label': pl.Float64\\}"):
        PolarsEngine().bind(undeclared).run(DF.lazy())
    declared = flow(frame_step(labelled, reads=["income"], writes={"label": pl.String}), name="p")
    out = streamed(PolarsEngine().bind(declared).run(DF.lazy()))
    assert_frame_equal(out, fused(declared))


def test_str_and_date_outputs_keep_their_dtype_when_a_batch_is_all_null():
    def band(income: float) -> str:
        return "top" if income > 95_000 else None

    def review_on(income: float) -> dt.date:
        return dt.date(2026, 1, 1) if income > 95_000 else None

    pipe = flow(band, review_on, name="p")
    out = streamed(PolarsEngine().bind(pipe).run(DF.lazy()))
    assert out.schema["band"] == pl.String and out.schema["review_on"] == pl.Date
    assert_frame_equal(out, fused(pipe))


def test_untyped_dict_output_is_refused_with_what_to_annotate():
    def details(income: float) -> dict:
        return {"income": income}

    with pytest.raises(EngineError, match="'details', written by p/details as dict.*TypedDict"):
        PolarsEngine().bind(flow(details, name="p")).run(DF.lazy())


def test_untyped_value_read_inside_its_segment_needs_no_dtype():
    def details(income: float) -> dict:
        return {"income": income}

    def doubled(details: dict) -> float:
        return details["income"] * 2

    pipe = flow(details, doubled, rank_income, name="p")
    assert_frame_equal(streamed(PolarsEngine().bind(pipe).run(DF.lazy())), fused(pipe))


def test_fallbacks_are_every_segments_fallbacks():
    def band(ratio: float) -> str:
        return "low" if ratio < 0.1 else "high"

    def grade(rank: float) -> str:
        return "top" if rank > 90 else "rest"

    pipe = flow(ratio, band, rank_income, grade, name="p")
    host = PolarsEngine().bind(pipe).fallbacks()
    assert sorted(host) == ["p/band", "p/grade"]
    assert host == FusedEngine().bind(pipe).fallbacks()


def test_dict_record_runs_on_the_segments_engine():
    record = {"income": 1000.0, "debt": 200.0}
    assert PolarsEngine().bind(PIPE).run(record) == FusedEngine().bind(PIPE).run(record)


def test_dataframe_input_returns_a_collected_dataframe():
    out = PolarsEngine().bind(PIPE).run(DF)
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, fused(PIPE))
    with pytest.raises(TypeError, match="dict record, a polars DataFrame or LazyFrame, not list"):
        PolarsEngine().bind(PIPE).run([1, 2])


def test_trace_and_session_are_refused_with_a_pointer_to_the_numba_engine():
    exe = PolarsEngine().bind(PIPE)
    for data in (DF.lazy(), {"income": 1000.0, "debt": 200.0}):
        with pytest.raises(EngineError, match="no trace; run the pipeline on the numba engine"):
            exe.run(data, trace=object())
    with pytest.raises(EngineError, match="no debug session; open one on the numba engine"):
        exe.session(DF)


def test_qualified_emit_is_refused():
    pipe = flow(score, step(score_read).relabel(writes={"score_read": "score"}), name="q").emit("score@q/score")
    with pytest.raises(EngineError, match="emits 'score@q/score'.*relabel"):
        PolarsEngine().bind(pipe)


class Boom(Exception):
    pass


def test_step_exception_type_and_note_survive_collect(tmp_path):
    def boom(income: float) -> float:
        if income > 50_000:
            raise Boom("too much")
        return income

    pipe = flow(boom, rank_income, name="p")
    with pytest.raises(Boom) as numba:
        fused(pipe)
    lf = PolarsEngine().bind(pipe).run(DF.lazy())
    for go in (lf.collect, lambda: streamed(lf), lambda: lf.sink_parquet(tmp_path / "out.parquet")):
        with pytest.raises(Boom) as host:
            go()
        assert host.value.__notes__ == numba.value.__notes__ == ["in step p/boom"]


def test_two_threads_share_one_polars_executable():
    exe = PolarsEngine().bind(PIPE)
    docs = [None, {"p": {"flag": {"min_rank": 50.0}}}]
    want = [fused(PIPE, params=d) for d in docs]
    start, got, errors = threading.Barrier(2), [[], []], []

    def work(i):
        try:
            start.wait()
            for _ in range(5):
                got[i].append(exe.run(DF.lazy(), docs[i]).collect())
        except Exception as e:  # noqa: BLE001 - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    for i in range(2):
        for out in got[i]:
            assert_frame_equal(out, want[i])
