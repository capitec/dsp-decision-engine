"""Prototype for experiment 05: step-ID source syntax and format.

Demonstrates the chosen syntax (a uniform optional `id=` keyword on every step
constructor, `@step(id=...)` as the function-step spelling) and exercises the
evolution scenarios named in the spike. The `decider` engine is not modified:
this checks the *syntax* decision, not a generator implementation.

Run: uv run python notes/vscode-redesign/experimentation/05-step-id/prototype.py
"""
from __future__ import annotations

import ast
import re
import secrets

ID_PATTERN = re.compile(r"[0-9a-f]{12}\Z")


def gen_id() -> str:
    """A 12-hex opaque token (48 bits), random so it never encodes name/path/position."""
    return secrets.token_hex(6)


# ---- 1. the syntax on each existing step form (before -> after) ----
FORMS = {
    "plain function (decorator spelling)": (
        "def debt_ratio(income: float, debt: float) -> float:\n    return debt / income\n",
        '@step(id="3f9a7c2e1b5d")\ndef debt_ratio(income: float, debt: float) -> float:\n    return debt / income\n',
    ),
    "@step(output=...)": (
        "@step(output=\"term_cap\")\ndef cap_by_income(term_cap: float) -> float:\n    return term_cap\n",
        '@step(output="term_cap", id="3f9a7c2e1b5d")\ndef cap_by_income(term_cap: float) -> float:\n    return term_cap\n',
    ),
    "@frame_step": (
        '@frame_step(reads=["client_id"], writes=["bureau_score"])\ndef join_bureau(df): ...\n',
        '@frame_step(reads=["client_id"], writes=["bureau_score"], id="3f9a7c2e1b5d")\ndef join_bureau(df): ...\n',
    ),
    "flow": (
        'term = flow(term_cap, cap_by_income, name="term")\n',
        'term = flow(term_cap, cap_by_income, name="term", id="3f9a7c2e1b5d")\n',
    ),
    "dag": (
        'p03 = dag(a, b, c, name="p03")\n',
        'p03 = dag(a, b, c, name="p03", id="3f9a7c2e1b5d")\n',
    ),
    "branch": (
        'by_sector = branch(cond, arm_a, arm_b, modifies=["cap"], name="by_sector")\n',
        'by_sector = branch(cond, arm_a, arm_b, modifies=["cap"], name="by_sector", id="3f9a7c2e1b5d")\n',
    ),
    "loop": (
        'repay = loop(unpaid, body, carries=["balance"], max_iterations=360, name="repay")\n',
        'repay = loop(unpaid, body, carries=["balance"], max_iterations=360, name="repay", id="3f9a7c2e1b5d")\n',
    ),
    "each": (
        'items = each("items", flow(heavy, name="item"), name="items")\n',
        'items = each("items", flow(heavy, name="item"), name="items", id="3f9a7c2e1b5d")\n',
    ),
    "optimise": (
        'best = optimise(count, evaluate, max_candidates=1024, name="best")\n',
        'best = optimise(count, evaluate, max_candidates=1024, name="best", id="3f9a7c2e1b5d")\n',
    ),
    "imported/reused function (call-site wrap)": (
        "pipeline = flow(deductions.statutory_deductions, offer)\n",
        'pipeline = flow(step(deductions.statutory_deductions, id="3f9a7c2e1b5d"), offer)\n',
    ),
}


def _parse_ok(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


# ---- 2. evolution scenarios over a minimal (id, name, module) model ----
Step = dict  # {"id": str, "name": str, "module": str}


def rename(step: Step, new_name: str) -> Step:
    out = dict(step)
    out["name"] = new_name
    return out


def extract(step: Step, new_module: str) -> Step:
    out = dict(step)
    out["module"] = new_module
    return out


def duplicate_detect(steps: list[Step]) -> list[str]:
    seen: dict[str, int] = {}
    dups = []
    for s in steps:
        seen[s["id"]] = seen.get(s["id"], 0) + 1
    return [sid for sid, n in seen.items() if n > 1]


def demo() -> None:
    # 1. every rendered form parses, and the id is well-formed
    for label, (before, after) in FORMS.items():
        assert _parse_ok(before), f"{label}: baseline must parse"
        assert _parse_ok(after), f"{label}: id form must parse"
    assert ID_PATTERN.match(gen_id()), "generated ids must be 12 hex chars"
    assert not ID_PATTERN.match("3f9a7c2e"), "short ids rejected"
    assert not ID_PATTERN.match("ratio:3f9a7c2e1b5d"), "prefixed ids rejected"

    # 2. rename / extract / reorder keep the id; copy/paste duplicates it
    base = {"id": "3f9a7c2e1b5d", "name": "debt_ratio", "module": "rules"}
    assert rename(base, "dti_ratio")["id"] == base["id"]
    assert extract(base, "core.ratios")["id"] == base["id"]
    steps = [base, dict(base, name="month_end", id="8b1d4f00a5c3"), dict(base, name="approved", id="e1d5a3c9b2f0")]
    reordered = steps[::-1]
    assert duplicate_detect(reordered) == [], "reordering must not collide ids"

    copied = steps + [dict(steps[0])]  # copy/paste: literal duplicate id
    assert duplicate_detect(copied) == ["3f9a7c2e1b5d"]

    # 3. independent branch edits: random ids never collide on merge
    branch_a = [gen_id() for _ in range(1000)]
    branch_b = [gen_id() for _ in range(1000)]
    assert len(set(branch_a) & set(branch_b)) == 0, "random ids must be disjoint across branches"

    print("prototype ok")


if __name__ == "__main__":
    demo()
