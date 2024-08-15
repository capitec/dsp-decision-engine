import pandas as pd
import numpy as np


# PD calcs
def log_odds_from_score(
    score_col: pd.Series, base_points: float, base_odds: float, pdo: float
) -> pd.Series:
    """Calculate log odds from a score."""
    factor = pdo / np.log(2)
    return (score_col - base_points) / factor + np.log(base_odds)


def probability_of_default_from_log_odds(log_odds_col: pd.Series) -> pd.Series:
    """Calculate PD from log odds."""
    exp_log_odds = np.exp(log_odds_col)
    return 1 / (1 + exp_log_odds)
