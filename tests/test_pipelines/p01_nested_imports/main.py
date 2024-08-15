from .util_module import do_calcs
from spockflow.core import initialize_spock_module


def sum_of_all(subtract_fn: int, add_fn: int, multiply_fn: int, divide_fn: int) -> int:
    return subtract_fn + add_fn + multiply_fn + divide_fn


initialize_spock_module(__name__, included_modules=[do_calcs])
