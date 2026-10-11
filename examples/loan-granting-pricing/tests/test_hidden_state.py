"""The steps of this product read no Python objects of their own: what they need is a param, a table or a document."""
import warnings

import pytest
from decikit import HiddenStateWarning, InterpretedEngine, flow

from loan_granting import eligibility, waterfall
from loan_granting.pipeline import build


@pytest.mark.parametrize("step", [eligibility.eligibility_gates, waterfall.cap_waterfall])
def test_a_step_reads_no_hidden_objects(step):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HiddenStateWarning)
        InterpretedEngine().bind(flow(step, name="t"))


def test_the_pipeline_reads_no_hidden_objects_of_its_own(documents):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InterpretedEngine().bind(build(**documents))
    # the embedded affordability assessment is that example's to fix
    own = [str(w.message) for w in caught if issubclass(w.category, HiddenStateWarning)
           and "/affordability_assessment/" not in str(w.message)]
    assert own == []
