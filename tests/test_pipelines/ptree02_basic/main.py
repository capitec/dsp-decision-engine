import pandas as pd
from spockflow.components.tree import Tree
from spockflow.components.common import Reject



tree = Tree()

@tree.condition(output=Reject(100))
def condition_a(a: pd.Series) -> pd.Series:
    return a > 10


@tree.condition()
def condition_b(b: pd.Series) -> pd.Series:
    return b > 10


@condition_b.condition(output=Reject(100))
def condition_c(c: pd.Series) -> pd.Series:
    return c > 10


@condition_b.condition(output=Reject(100))
def condition_c(c: pd.Series) -> pd.Series:
    return c > 10