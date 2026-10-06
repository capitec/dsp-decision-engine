"""Id checks: duplicates of one durable id."""
from decider import flow, step
from decider.lab import checks


@step(output="a", id="0123abcdef45")
def first(x: float) -> float:
    return x


@step(output="b", id="0123abcdef45")
def copied(a: float) -> float:
    return a


@step(output="c", id="0123abcdef46")
def other(b: float) -> float:
    return b


def test_two_steps_with_one_id_are_an_error_at_the_second():
    report = checks.run(flow(first, copied, name="f"), ["duplicate_ids"])
    (f,) = report.findings
    assert (f.code, f.step.path, f.detail["other"]) == ("DCD402", "f/copied", "f/first")
    assert not report.ok


def test_distinct_ids_are_quiet():
    assert not checks.run(flow(first, other, name="f"), ["duplicate_ids"]).findings
