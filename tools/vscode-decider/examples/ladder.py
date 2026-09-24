"""A plain step that loops over a parameter table: a rate ladder by bureau score."""
from decider import Table, flow, param_table


def rate(score: int, ladder: Table = param_table({"floor": int, "rate": float},
                                                  default=[{"floor": 0, "rate": 0.2}, {"floor": 650, "rate": 0.12}])) -> float:
    r = 0.0
    for i in range(len(ladder.floor)):
        if score >= ladder.floor[i]:
            r = ladder.rate[i]
    return r


pricing = flow(rate, name="pricing")

SAMPLE = [{"client_id": 1, "score": 550}, {"client_id": 2, "score": 720}]
