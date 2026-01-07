# Where to begin
1. Create a hamilton dag with polars and execute
2. Decorator to use both polars and pandas so can we create a function in hamilton that is a pandas function that executes

``` module```
def func1(df: pl.Polars) -> pl.Polars

@convert_types
def func2(func1: pd.DF) -> pd.DF

@convert_types
def func3(func2: pl.Polars) -> pl.Polars

3. Config for components
4. Higherlevel hamilton to build build dags
