from numba import njit


@njit(cache=True)
def driver(x):
    if x > 0.0405:
        return x * 2.0
    else:
        return x * 3.0
