
import pandas as pd
import numpy as np
# from io import StringIO
# df = pd.read_csv("../../progs/data/tree.csv")
# # %%
# df_scorecard = pd.DataFrame({
#     "x_MIN": [1,None,3,4,5,6,7],
#     "x_MAX": [2,3,4,None,6,7,8],
#     "x_IN": [[1,2,3],None,[1,2,3],[1,2,3],[1,2,3],[1,2,3],None],
#     "x_score_OUT": [5,6,7,8,9,10,11]
# })
# test = pd.read_csv(StringIO(df_scorecard.to_csv()))

def score_table(df_score_table: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    mask = np.array([[True]*len(df_score_table)]*len(df))
    for col in df_score_table.columns:
        if col.endswith("_MIN"):
            criteria_mask = df[col[:-4]].values[None] >= df_score_table[col].values[:,None]
            criteria_mask[df_score_table[col].isnull()]=True
        if col.endswith("_MAX"):
            criteria_mask = df[col[:-4]].values[None] < df_score_table[col].values[:,None]
            criteria_mask[df_score_table[col].isnull()]=True
        if col.endswith("_IN"):
            df_values = df[col[:-3]].values
            criteria_mask = np.apply_along_axis( # TODO maybe write this as a numba function
                lambda v: np.isin(df_values,eval(v[0])) if not pd.isnull(v[0]) else [True]*len(df_values), 
                0, df_score_table[col].values[None]
            )
        else:
            continue
        mask &= criteria_mask
    value_idx = np.argmax(mask,axis=1)
    assert all(mask[range(mask.shape[0]),value_idx]), "One or more columns didnt match any criteria"
    out_cols = [c for c in df_score_table if c.endswith("_OUT")]
    out_df = df_score_table[out_cols].iloc[value_idx]
    return out_df.rename(columns={c:c[:-4] for c in out_df.columns}).set_index(df.index)

