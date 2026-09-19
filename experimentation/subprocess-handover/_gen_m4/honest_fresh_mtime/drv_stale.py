# decider2 generated driver
from numba import njit


@njit(cache=True)
def driver(a0, a1, a2, a3, a4, a5, a6, a7):
    acc = 0.0
    if a0 > 0.5:
        acc += a0 * 1.0
    else:
        acc -= a0 * 1.0 * 0.5
    if a1 > 0.5:
        acc += a1 * 1.001
    else:
        acc -= a1 * 1.001 * 0.9
    if a2 > 0.5:
        acc += a2 * 1.002
    else:
        acc -= a2 * 1.002 * 0.5
    if a3 > 0.5:
        acc += a3 * 1.003
    else:
        acc -= a3 * 1.003 * 0.5
    if a4 > 0.5:
        acc += a4 * 1.004
    else:
        acc -= a4 * 1.004 * 0.5
    if a5 > 0.5:
        acc += a5 * 1.005
    else:
        acc -= a5 * 1.005 * 0.5
    if a6 > 0.5:
        acc += a6 * 1.006
    else:
        acc -= a6 * 1.006 * 0.5
    if a7 > 0.5:
        acc += a7 * 1.0
    else:
        acc -= a7 * 1.0 * 0.5
    return acc
