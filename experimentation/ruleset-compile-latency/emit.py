"""Emitter: turns a rule-set spec into numba-compilable Python source.

Vocabulary mirrored from decider/modules/rules/ (flat_rules/nodes.py,
common/nodes/{operators,cases,composite,unary}.py):

  13 unary operators : <= < == > >= != between isin string_match
                       is_null is_not_null is_true is_false
  3 cases variants   : ranges / string_match / isin  (multi-way + otherwise)
  CompositeRule      : AND / OR / NOT over 2-4 conditions, with then/otherwise
  LeafRule           : terminal result_idx
  FlatRuleTree       : list of independent rule trees
  PrioritizationMode : first_match (early exit) | all (evaluate everything)

Deliberate simplification, stated up front: string features are represented as
int32 category codes, because numba nopython mode has no array-of-strings dtype.
`string_match` therefore emits the same shape as `isin` (a disjunction of code
equalities).  This is what a real numba boundary would have to do anyway (doc 05
§1.5 admissibility), but it means this harness does NOT measure the compile cost
of numba's unicode string ops.
"""

import random

# feature pools available to emitted kernels
N_FLOAT = 16   # numeric features           -> f0..f15   (float64[::1])
N_CODE = 6     # categorical / string codes -> c0..c5    (int64[::1])
N_BOOL = 3     # boolean flags              -> b0..b2    (boolean[::1])

UNARY_OPS = [
    "le", "lt", "eq", "gt", "ge", "ne",
    "between", "isin", "string_match",
    "is_null", "is_not_null", "is_true", "is_false",
]


def _f(rng):
    return f"f{rng.randrange(N_FLOAT)}[i]"


def _c(rng):
    return f"c{rng.randrange(N_CODE)}[i]"


def _b(rng):
    return f"b{rng.randrange(N_BOOL)}[i]"


def unary_clause(rng, op=None):
    """One unary operator rendered as a numba boolean expression."""
    op = op or rng.choice(UNARY_OPS)
    if op in ("le", "lt", "eq", "gt", "ge", "ne"):
        sym = {"le": "<=", "lt": "<", "eq": "==", "gt": ">", "ge": ">=", "ne": "!="}[op]
        if op in ("eq", "ne"):
            # equality on a numeric feature is realistically against a code
            return f"{_c(rng)} {sym} {rng.randrange(10)}"
        return f"{_f(rng)} {sym} {rng.random():.4f}"
    if op == "between":
        lo = rng.random() * 0.5
        hi = lo + 0.1 + rng.random() * 0.4
        v = _f(rng)
        return f"({v} >= {lo:.4f} and {v} <= {hi:.4f})"
    if op in ("isin", "string_match"):
        v = _c(rng)
        vals = rng.sample(range(10), rng.randrange(2, 5))
        return "(" + " or ".join(f"{v} == {k}" for k in vals) + ")"
    if op == "is_null":
        v = _f(rng)
        return f"{v} != {v}"          # NaN sentinel
    if op == "is_not_null":
        v = _f(rng)
        return f"{v} == {v}"
    if op == "is_true":
        return _b(rng)
    if op == "is_false":
        return f"not {_b(rng)}"
    raise ValueError(op)


def composite_expr(rng, n_clauses=None):
    """CompositeRule: AND/OR over 2-4 unary clauses, sometimes NOT-wrapped."""
    n = n_clauses or rng.randrange(2, 5)
    joiner = " and " if rng.random() < 0.6 else " or "
    inner = joiner.join(unary_clause(rng) for _ in range(n))
    if rng.random() < 0.15:
        return f"not ({inner})"
    return f"({inner})"


# no-match decisions draw from a SEPARATE rng stream so that changing p_nomatch
# leaves the emitted tree shape (and therefore the line count and compile time)
# identical -- only which leaves are -1 changes.
_NM = {"rng": None, "p": 0.0}


def _leaf(rng, target, out, p_nomatch=0.0):
    """LeafRule. result_idx=-1 is the flat_rules 'no match' sentinel -- see
    WithUnaryBranches._get_then_rule(), which defaults to LeafRule(result_idx=-1).
    p_nomatch controls how often a leaf declines to match, which is what makes
    first_match actually fall through to the next rule."""
    idx = rng.randrange(1, 900)   # always drawn, so shape rng stays in lockstep
    if _NM["p"] > 0.0 and _NM["rng"].random() < _NM["p"]:
        idx = -1
    out.append(f"{target}res = {idx}")


def emit_subtree(rng, indent, out, depth, p_nomatch=0.0):
    """A `then`/`otherwise` side: leaf, nested unary, or a cases node."""
    pad = "    " * indent
    if depth <= 0:
        _leaf(rng, pad, out, p_nomatch)
        return
    roll = rng.random()
    if roll < 0.40:
        _leaf(rng, pad, out, p_nomatch)
    elif roll < 0.70:
        # nested UnaryRule with then/otherwise
        out.append(f"{pad}if {unary_clause(rng)}:")
        emit_subtree(rng, indent + 1, out, depth - 1, p_nomatch)
        out.append(f"{pad}else:")
        emit_subtree(rng, indent + 1, out, depth - 1, p_nomatch)
    else:
        # CasesRule: ranges / isin / string_match, 3-5 branches + otherwise
        variant = rng.choice(["ranges", "isin", "string_match"])
        n_br = rng.randrange(3, 6)
        if variant == "ranges":
            v = _f(rng)
            edges = sorted(round(rng.random(), 4) for _ in range(n_br - 1))
            for k, e in enumerate(edges):
                kw = "if" if k == 0 else "elif"
                out.append(f"{pad}{kw} {v} < {e}:")
                emit_subtree(rng, indent + 1, out, depth - 1, p_nomatch)
            out.append(f"{pad}else:")
            emit_subtree(rng, indent + 1, out, depth - 1, p_nomatch)
        else:
            v = _c(rng)
            pool = list(range(10))
            rng.shuffle(pool)
            for k in range(n_br - 1):
                grp = pool[k * 2:k * 2 + 2] or [k]
                test = " or ".join(f"{v} == {g}" for g in grp)
                kw = "if" if k == 0 else "elif"
                out.append(f"{pad}{kw} {test}:")
                emit_subtree(rng, indent + 1, out, depth - 1, p_nomatch)
            out.append(f"{pad}else:")
            emit_subtree(rng, indent + 1, out, depth - 1, p_nomatch)


def emit_rule_body(rng, indent, out, depth, p_nomatch=0.0):
    """One RuleRoot: composite condition -> then subtree / otherwise subtree."""
    pad = "    " * indent
    out.append(f"{pad}if {composite_expr(rng)}:")
    emit_subtree(rng, indent + 1, out, depth, p_nomatch)
    out.append(f"{pad}else:")
    emit_subtree(rng, indent + 1, out, max(depth - 1, 0), p_nomatch)


def emit_ruleset(n_rules, mode, seed=0, depth=2, fname="ruleset", p_nomatch=0.0):
    """Emit a full FlatRuleTree kernel. mode in {'first_match', 'all'}."""
    assert mode in ("first_match", "all")
    rng = random.Random(seed)
    _NM["rng"] = random.Random(seed ^ 0xABCDEF)
    _NM["p"] = p_nomatch
    args = (
        [f"f{k}" for k in range(N_FLOAT)]
        + [f"c{k}" for k in range(N_CODE)]
        + [f"b{k}" for k in range(N_BOOL)]
        + ["out"]
    )
    out = []
    out.append(f"def {fname}({', '.join(args)}):")
    out.append("    n = out.shape[0]")
    out.append("    for i in range(n):")
    if mode == "first_match":
        out.append("        res = -1")
        for r in range(n_rules):
            out.append(f"        if res == -1:")
            emit_rule_body(rng, 3, out, depth, p_nomatch)
        out.append("        out[i] = res")
    else:
        for r in range(n_rules):
            out.append("        res = -1")
            emit_rule_body(rng, 2, out, depth, p_nomatch)
            out.append(f"        out[i, {r}] = res")
    return "\n".join(out) + "\n"


def count_lines(src):
    return sum(1 for ln in src.splitlines() if ln.strip())
