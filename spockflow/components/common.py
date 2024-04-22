import pandas as pd


def Reject(code: int, description: str, soft_reject: bool=False) -> pd.DataFrame:
    df = pd.DataFrame(
        [dict(code = code, description=description, soft_reject=soft_reject)]
    )
    df._rule_engine_internal_prop_name_ = f"REJECT Code={code}"
    return df