"""The steps read no Python objects of their own: what they need is a param, a table, a document or a plain tuple."""
import warnings

from decikit import Engine, HiddenStateWarning

from consolidation.pipeline import build


def test_the_pipeline_reads_no_hidden_objects(documents):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Engine().bind(build(*documents), mode="interpreted")
    assert [str(w.message) for w in caught if issubclass(w.category, HiddenStateWarning)] == []
