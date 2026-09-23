import pytest

from decider.testing import assert_equivalent


@pytest.fixture
def run():
    """`run(step, df, params=None)`: the output, after checking every mode agrees with the Python matcher.

    It checks `run()` in every mode, `score()` of every row and a resumed session.
    """
    return assert_equivalent
