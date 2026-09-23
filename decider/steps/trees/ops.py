"""Opcodes for the six threshold comparisons, shared by the tree compilers."""

LT, LE, EQ, GT, GE, NE = range(6)

OPCODE = {"<": LT, "<=": LE, "==": EQ, ">": GT, ">=": GE, "!=": NE}
