"""P10 affordability chain — the five-line arithmetic. Spec 5.11. One
arithmetic, four evidence modes (modes.py), zero `if`s here.

`basis="inherited"` on the P10 envelope (phases/__init__.py) means every
multi-basis read below resolves to the invocation's AMBIENT basis
(values/bases.py) — nobody in this file types `actual()` or `hypothetical()`.
The same five lines run product-neutrally before P11, per-product after it,
and up to three more times inside the loop (loops/l1_consolidation.py), each
time under a different ambient basis the caller sets, never this file.
"""

from __future__ import annotations

from decider2 import module, Table, param

class ResidualFloorTable(Table):
    key: int              # dependants_count, 0..6+
    floor: float

class RatioCeilingTable(Table):
    key: tuple[int, int]  # (risk_grade, product_code)
    ratio: float

class BufferGridTable(Table):
    key: tuple[int, int, int]   # (risk_grade, product_code, channel_class) — 288 cells
    buffer: float

def discretionary_income(net_monthly_income: float, living_expenses: float,
                         existing_obligations: float) -> float:
    pass  # net_monthly_income - living_expenses - existing_obligations

def capacity(discretionary_income: float, dependants_count: int,
            residual_floors: ResidualFloorTable) -> float:
    pass  # discretionary_income - residual_floors[dependants_count]

def ratio_ceiling(risk_grade: int, product_code: int, net_monthly_income: float,
                  ratio_ceilings: RatioCeilingTable) -> float:
    pass  # ratio_ceilings[(risk_grade, product_code)] * net_monthly_income

def pre_buffer(capacity: float, ratio_ceiling: float) -> float:
    pass  # min(capacity, ratio_ceiling)

def max_affordable_instalment(pre_buffer: float, risk_grade: int, product_code: int,
                              channel_class: int, buffer_grid: BufferGridTable) -> float:
    """Site-attributed (values/register.py `site_attribution_required=True`):
    the caller's `loop_pass_index` and site tag land on this value automatically,
    never typed here."""
    pass  # pre_buffer * (1 - buffer_grid[(risk_grade, product_code, channel_class)])

Chain = module(discretionary_income, capacity, ratio_ceiling, pre_buffer,
               max_affordable_instalment, name="chain")
