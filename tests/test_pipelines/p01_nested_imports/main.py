from util_module import do_calcs

DIRECT_MODULES = [do_calcs]

def sum_of_all(subtract_fn: int, add_fn: int, multiply_fn: int, divide_fn: int) -> int:
    return subtract_fn+add_fn+multiply_fn+divide_fn

OUTPUTS = [sum_of_all]