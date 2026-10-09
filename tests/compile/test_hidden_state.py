"""`hidden_state` lists the objects a step reads that a config document, a diff or a kernel cannot see."""
import datetime
import enum
import statistics
import types
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from decider import step
from decider._engine.compile import fingerprint, hidden_state


class Register:
    limit = 3.0


class Tier(enum.Enum):
    GOLD = 1


REGISTER = Register()
RATES = {"gold": 1.0}
CODES = (1, 2)
NAMES = frozenset({"a"})
WHEN = datetime.date(2024, 1, 1)
MONEY = Decimal("1.5")
ARRAY = np.arange(3.0)
FRAME = pl.DataFrame({"a": [1]})
HELPERS = types.ModuleType("helpers")
HELPERS.LIMIT, HELPERS.SHARED = 5.0, Register()


def reads_a_global_object(x: float) -> float:
    return x + REGISTER.limit


def reads_a_dict(x: float) -> float:
    return x + RATES["gold"]


def reads_data(x: float) -> float:
    return x + CODES[0] + len(NAMES) + WHEN.year + float(MONEY) + ARRAY[0] + FRAME.height + Tier.GOLD.value + abs(x)


def reads_a_library(x: float) -> float:
    return statistics.mean([x, 1.0])


def reads_module_data(x: float) -> float:
    return x + HELPERS.LIMIT


def reads_a_module_object(x: float) -> float:
    return x + HELPERS.SHARED.limit


def closing_over(register):
    def capped(x: float) -> float:
        return min(x, register.limit)

    return capped


def test_a_global_object_is_named_with_its_class_and_where_it_is_read():
    [(name, kind, type_name, site)] = hidden_state(reads_a_global_object)
    assert (name, kind, type_name) == ("REGISTER", "object", "Register")
    assert site.startswith("module global in ") and site.endswith(f"test_hidden_state.py:{reads_a_global_object.__code__.co_firstlineno}")


def test_a_closure_variable_is_named_with_the_function_that_closes_over_it():
    [(name, kind, type_name, site)] = hidden_state(closing_over(Register()))
    assert (name, kind, type_name) == ("register", "object", "Register")
    assert site.startswith("closure variable of closing_over.<locals>.capped, ")


def test_a_dict_global_is_flagged_as_a_container():
    assert [h[:3] for h in hidden_state(reads_a_dict)] == [("RATES", "container", "dict")]


def test_a_dict_of_objects_is_one_finding_not_one_per_object():
    holder = {"a": Register(), "b": Register()}

    def step_(x: float) -> float:
        return x + len(holder)

    assert [h[:3] for h in hidden_state(step_)] == [("holder", "container", "dict")]


def test_values_functions_modules_enums_arrays_and_frames_are_not_hidden_state():
    assert hidden_state(reads_data) == []
    assert hidden_state(reads_a_library) == []


def test_a_module_attribute_read_as_module_dot_name_is_named_that_way():
    assert hidden_state(reads_module_data) == []
    assert [h[:3] for h in hidden_state(reads_a_module_object)] == [("HELPERS.SHARED", "object", "Register")]


def test_a_step_held_in_a_closure_is_not_hidden_state():
    inner = step(reads_data)

    def outer(x: float) -> float:
        return x + (inner is None)

    assert hidden_state(outer) == []


@pytest.mark.parametrize("fn, hidden", [
    (reads_a_global_object, True), (closing_over(Register()), True), (reads_a_module_object, True),
    (reads_data, False), (reads_a_library, False), (reads_module_data, False),
], ids=lambda v: getattr(v, "__name__", v))
def test_it_flags_a_step_exactly_when_its_strict_key_cannot_be_built(fn, hidden):
    def strict_key_fails():
        try:
            fingerprint(fn, strict=True)
        except ValueError:
            return True
        return False

    assert bool(hidden_state(fn)) == strict_key_fails() == hidden
