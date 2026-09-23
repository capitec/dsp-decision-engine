import polars as pl

from decider import flow, step


def test_a_nullable_string_input_reaches_the_step_as_none():
    @step
    def label(code: str | None) -> str:
        return "none" if code is None else code.upper()

    out = flow(label).run(pl.DataFrame({"code": ["a", None, "c"]}))
    assert out["label"].to_list() == ["A", "none", "C"]
