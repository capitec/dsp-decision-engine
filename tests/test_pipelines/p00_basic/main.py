def a() -> int:
    return 1


def b(a: int) -> float:
    return 3.14 * a


def c(q: int, a: int, b: float) -> float:
    return q + a + b


OUTPUTS = [a, c]
