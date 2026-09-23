from decider import param
from decider.engine.params import NodeParams, Status, harvest, validate_node


def sector_rate(sector: str, private: str = param("private"),
                groups: list[str] = param(["gov", "public"]), rate: float = param(0.9)) -> float:
    return rate if sector == private else 1.0


def test_a_str_param_is_still_a_plain_str_default():
    assert sector_rate("private") == 0.9
    assert sector_rate("public") == 1.0


def test_str_and_list_of_str_params_validate_into_the_bundle_as_strings():
    node = NodeParams("sector_rate", harvest(sector_rate)[1])
    assert node.defaults == ("private", ["gov", "public"], 0.9)
    result = validate_node(node, {"sector_rate": {"private": "government", "groups": ["a"]}})
    assert result.bundle == ("government", ["a"], 0.9)
    assert type(result.bundle) is type(node.defaults)


def test_a_non_string_value_for_a_str_param_is_invalid():
    node = NodeParams("sector_rate", harvest(sector_rate)[1])
    assert validate_node(node, {"sector_rate": {"private": 3}}).status is Status.INVALID
