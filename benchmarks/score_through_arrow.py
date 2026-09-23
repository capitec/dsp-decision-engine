"""score(dict) today against score(dict) through a one-row polars frame and the Arrow shim.

    uv run python benchmarks/score_through_arrow.py [flagship|wide|tree ...]

Prints p50/p99 of each path, then the median cost of each piece of the Arrow path.
"""
import gc
import sys
import time

import polars as pl

sys.path.insert(0, "benchmarks")
import zero_copy_spike as Z  # noqa: E402
from decider.engine import Engine  # noqa: E402
from decider.engine.run.state import State  # noqa: E402


def through_arrow(exe, frame_of):
    # One input implementation: the record becomes a one-row frame and is read like a batch.
    plan, runner = exe.plan, exe.runner

    def score(record):
        state = State.from_frame(plan, frame_of(record), 1)
        run = exe._params(None, 1)
        for _ in runner.iterate(plan, state, run):
            pass
        out = {k: x for k, x in record.items() if k not in exe._hidden}
        for k, v in exe._results:
            values, valid = state.read(v)
            out[k] = None if valid is not None and not valid[0] else values.tolist()[0]
        return out

    return score


FRAMES = {
    "pl.DataFrame([record])": lambda r: pl.DataFrame([r]),
    "pl.DataFrame({k: [v]})": lambda r: pl.DataFrame({k: [v] for k, v in r.items()}),
}


def stats(f, arg, calls=20000):
    for _ in range(500):
        f(arg)
    gc.collect()
    s = []
    for _ in range(calls):
        t = time.perf_counter()
        f(arg)
        s.append(time.perf_counter() - t)
    s.sort()
    return s[len(s) // 2] * 1e6, s[int(len(s) * 0.99)] * 1e6


def main(names):
    for name in names:
        pipeline, df = Z.BUILDERS[name](1000)
        exe = Engine().bind(pipeline, mode="fused")
        record = df.row(0, named=True)
        exe.run(df)
        want = exe.score(record)
        print(f"{name} ({len(record)} inputs)")
        p50, p99 = stats(exe.score, record)
        print(f"  {'score() today':<34}p50 {p50:6.1f}  p99 {p99:6.1f} us")
        p50, p99 = stats(exe.run, df.head(1))
        print(f"  {'run(one-row frame)':<34}p50 {p50:6.1f}  p99 {p99:6.1f} us")
        for label, frame_of in FRAMES.items():
            f = through_arrow(exe, frame_of)
            assert f(record) == want, (f(record), want)
            p50, p99 = stats(f, record)
            print(f"  {'via ' + label:<34}p50 {p50:6.1f}  p99 {p99:6.1f} us")
        # Pieces of the Arrow path.
        frame = pl.DataFrame([record])
        state = State.from_frame(exe.plan, frame, 1)
        run = exe._params(None, 1)

        def iterate(_):
            st = State.from_frame(exe.plan, frame, 1)
            for _ in exe.runner.iterate(exe.plan, st, run):
                pass

        pieces = {
            "build frame pl.DataFrame([r])": (lambda r: pl.DataFrame([r]), record),
            "export+read (State.from_frame)": (lambda f: State.from_frame(exe.plan, f, 1), frame),
            "from_frame + kernels": (iterate, None),
            "params": (lambda _: exe._params(None, 1), None),
        }
        for label, (f, arg) in pieces.items():
            p50, _ = stats(f, arg, 5000)
            print(f"    {label:<32}{p50:6.1f} us")


if __name__ == "__main__":
    main(sys.argv[1:] or list(Z.BUILDERS))
