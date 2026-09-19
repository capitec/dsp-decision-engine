"""L-shape driver generator: exactly ONE occurrence of the threshold literal
in the whole function body, so a same-byte-length edit to it cannot collide
with any other constant's co_consts slot (unlike K's gen_driver.py, where
0.5 is shared across every branch)."""


def gen_source(threshold: str) -> str:
    return (
        "from numba import njit\n\n\n"
        "@njit(cache=True)\n"
        "def driver(x):\n"
        f"    if x > {threshold}:\n"
        "        return x * 2.0\n"
        "    else:\n"
        "        return x * 3.0\n"
    )
