import pandas as pd


def test_1(input_1: pd.Series, input_2: pd.Series) -> pd.Series:
    return input_1


def test_2(input_2: pd.Series) -> pd.Series:
    return input_2

def test_3(test_1: pd.Series, test_2: pd.Series) -> pd.Series:
    return test_1

def test_4(test_2: pd.Series) -> pd.Series:
    return test_2

def test_5(test_4: pd.Series) -> pd.Series:
    return test_4

def test_6(test_4: pd.Series, test_5: pd.Series) -> pd.Series:
    return test_4