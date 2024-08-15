import typing


def safe_update(dict_new: dict, dict_old: dict):
    for k, v in dict_old.items():
        if k in dict_new:
            assert (
                dict_new[k] is dict_old[k]
            ), f"Value {k} used multiple times with different input values"
        else:
            dict_new[k] = v


def get_name(value, value_name: typing.Optional[str]):
    import types
    import pandas as pd

    if value_name is not None:
        return value_name
    if isinstance(value, pd.Series) and value.name is not None:
        return value.name
    if isinstance(value, pd.DataFrame) and value.attrs.get("name") is not None:
        return value.attrs.get("name")
    if isinstance(value, types.FunctionType):
        return value.__name__
    internal_name = getattr(value, "_rule_engine_internal_prop_name_", None)
    if internal_name is not None:
        return internal_name
    internal_name = getattr(value, "name", None)
    if internal_name is not None:
        return internal_name
    raise ValueError(
        f"Could not infer property name. Please manually provide a name for the value {value}"
    )


def index_config_path(config: typing.Dict[str, typing.Any], path: str):
    curr = config
    if path == "":
        return curr
    for el in path.split("."):
        # if el == "": continue
        curr = curr[el]
    return curr
