import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def driver(inp, coef_f8, out_f8, out_i8, out_b):
    n = inp.shape[0]
    for i in range(n):
        a0 = inp[i].i_f8_0
        a1 = inp[i].i_f8_1
        a2 = inp[i].i_f8_2
        a3 = inp[i].i_f8_3
        a4 = inp[i].i_f8_4
        b0 = inp[i].i_i8_0

        for j in range(380):
            out_f8[i, j] = coef_f8[j] * a0 + a1 - a2 * 0.001
        for j in range(158):
            out_i8[i, j] = b0 + j
        for j in range(95):
            out_b[i, j] = (a3 + j * 0.0001) > a4
