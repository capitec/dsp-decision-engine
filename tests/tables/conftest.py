import pytest

from decikit.lab.testing import assert_equivalent


@pytest.fixture
def run():
    """`run(step, df, params=None)`: the output, after checking every mode agrees with the Python matcher.

    It checks `run()` in every mode, `run()` of every row as a record and a resumed session.
    """
    return assert_equivalent
