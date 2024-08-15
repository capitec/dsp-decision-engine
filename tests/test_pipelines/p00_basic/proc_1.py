from transforms import *

def b(a: int):
    return a*2

def c(b:int, trf_a: int) -> int:
    return b*trf_a

OUTPUTS = [c]