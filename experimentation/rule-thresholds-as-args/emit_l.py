"""Emitter for EXPERIMENT L -- two source forms of the SAME rule set.

Simplification, stated up front (time-budget choice, not hidden): rules here
are a single flat AND-of-comparisons composite per rule (first_match, early
exit), matching the TOP layer of experimentation/ruleset-compile-latency/emit.py's
tree (composite_expr + a leaf) but WITHOUT that harness's nested cases/unary
sub-trees. That top layer is exactly the layer that carries numeric
thresholds -- the thing a business-user UI edits -- so it is the right slice
for this question, but it means line counts here are not directly comparable
to G's table (fewer lines per rule). Both forms below are emitted from the
IDENTICAL rule spec (same feature indices, same operators, same threshold
VALUES) so every difference measured is attributable only to "literal vs
argument", never to a shape difference.

  literal form: `if f3[i] <= 0.4231 and f7[i] > 0.1187: ...`
  args form:    `if f3[i] <= th[0] and f7[i] > th[1]: ...`
  args+mask:    `if mask[r] and f3[i] <= th[0] and f7[i] > th[1]: ...`
"""

from __future__ import annotations

import random

N_FLOAT = 16  # feature pool, mirrors ruleset-compile-latency/emit.py
OPSYM = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">"}


def gen_rule_spec(n_rules: int, n_clauses: int = 3, seed: int = 0):
    """[(feat:int, op:str, thresh:float), ...] per rule -- the SHAPE and the
    VALUES, shared by both emitters below."""
    rng = random.Random(seed)
    rules = []
    for _ in range(n_rules):
        clauses = []
        for _ in range(n_clauses):
            feat = rng.randrange(N_FLOAT)
            op = rng.choice(list(OPSYM))
            thresh = round(rng.random(), 4)
            clauses.append((feat, op, thresh))
        rules.append(clauses)
    return rules


def _sig(extra_args=()):
    feats = ", ".join(f"f{k}" for k in range(N_FLOAT))
    tail = "".join(f", {a}" for a in extra_args)
    return f"{feats}, out{tail}"


def emit_literal(rules, fname="driver") -> str:
    lines = [f"def {fname}({_sig()}):",
              "    n = out.shape[0]",
              "    for i in range(n):",
              "        res = -1"]
    for r, clauses in enumerate(rules):
        cond = " and ".join(f"f{feat}[i] {OPSYM[op]} {thresh}" for feat, op, thresh in clauses)
        lines.append(f"        if res == -1 and ({cond}):")
        lines.append(f"            res = {r}")
    lines.append("        out[i] = res")
    return "\n".join(lines) + "\n"


def emit_args(rules, n_clauses: int, fname="driver") -> str:
    """Same shape as emit_literal, but every threshold is a read from a flat
    `th` array argument -- index = r * n_clauses + c. No threshold value is
    emitted into source at all."""
    lines = [f"def {fname}({_sig(['th'])}):",
              "    n = out.shape[0]",
              "    for i in range(n):",
              "        res = -1"]
    for r, clauses in enumerate(rules):
        parts = []
        for c, (feat, op, _thresh) in enumerate(clauses):
            idx = r * n_clauses + c
            parts.append(f"f{feat}[i] {OPSYM[op]} th[{idx}]")
        cond = " and ".join(parts)
        lines.append(f"        if res == -1 and ({cond}):")
        lines.append(f"            res = {r}")
    lines.append("        out[i] = res")
    return "\n".join(lines) + "\n"


def emit_args_masked(rules, n_clauses: int, fname="driver") -> str:
    """args form + a per-rule bool `mask` argument: disabling a rule is now
    ALSO a value, not a source edit."""
    lines = [f"def {fname}({_sig(['th', 'mask'])}):",
              "    n = out.shape[0]",
              "    for i in range(n):",
              "        res = -1"]
    for r, clauses in enumerate(rules):
        parts = [f"mask[{r}]"]
        for c, (feat, op, _thresh) in enumerate(clauses):
            idx = r * n_clauses + c
            parts.append(f"f{feat}[i] {OPSYM[op]} th[{idx}]")
        cond = " and ".join(parts)
        lines.append(f"        if res == -1 and ({cond}):")
        lines.append(f"            res = {r}")
    lines.append("        out[i] = res")
    return "\n".join(lines) + "\n"


def thresholds_array(rules, n_clauses: int):
    import numpy as np
    flat = []
    for clauses in rules:
        for _feat, _op, thresh in clauses:
            flat.append(thresh)
    assert len(flat) == len(rules) * n_clauses
    return np.array(flat, dtype=np.float64)


def count_lines(src: str) -> int:
    return sum(1 for ln in src.splitlines() if ln.strip())


if __name__ == "__main__":
    spec = gen_rule_spec(3, seed=0)
    print(emit_literal(spec))
    print("---")
    print(emit_args(spec, n_clauses=3))
