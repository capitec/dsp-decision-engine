"""The steps of this product read no Python objects of their own: what they need is a param, a table or a document."""
import warnings

import pytest
from decider import Engine, HiddenStateWarning, flow

from loan_granting import eligibility, waterfall


@pytest.mark.parametrize("step", [eligibility.eligibility_gates, waterfall.cap_waterfall])
def test_a_step_reads_no_hidden_objects(step):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HiddenStateWarning)
        Engine().bind(flow(step, name="t"), mode="interpreted")
