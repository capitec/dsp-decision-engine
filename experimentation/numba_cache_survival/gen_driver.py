"""Deterministic generator for a numba driver source file.

Stands in for decider2's build-time codegen: a single njit'd driver with many
scalar arguments and one branch per argument, so compile cost is dominated by
the generated glue rather than by any one step.

Byte-for-byte determinism is the point: gen_source(n, comment) must return the
exact same bytes for the same inputs, on any machine, in any process.
"""


def gen_source(n_args: int, comment: str = "decider2 generated driver") -> str:
    a = [f"a{i}" for i in range(n_args)]
    out = [
        f"# {comment}",
        "from numba import njit",
        "",
        "",
        "@njit(cache=True)",
        f"def driver({', '.join(a)}):",
        "    acc = 0.0",
    ]
    for i, name in enumerate(a):
        # One real branch per argument: short-circuiting glue, as doc 05 sec 4.3
        # describes for Branch nodes. Constants vary per index so LLVM cannot
        # collapse the blocks into one.
        k = 1.0 + (i % 7) / 1000.0
        out.append(f"    if {name} > 0.5:")
        out.append(f"        acc += {name} * {k!r}")
        out.append("    else:")
        out.append(f"        acc -= {name} * {k!r} * 0.5")
    out.append("    return acc")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    sys.stdout.write(gen_source(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
