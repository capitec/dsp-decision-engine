import typing
import pandas as pd
from lib.builtin_funcs.scoretable_impl import score_table
from lib.annotators import dangerously_skip_type_validation
from lib.creators import parametrized_input_output

@dangerously_skip_type_validation
@parametrized_input_output()
def run_score_table(score_table_df: pd.DataFrame, **kwargs: typing.Dict[str, pd.Series]) -> pd.DataFrame:
    return score_table(score_table_df, pd.DataFrame(kwargs))
