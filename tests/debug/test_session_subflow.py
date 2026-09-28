"""Stepping into an `each` child: step_into reaches the child's steps and value/set work on them."""
import polars as pl

from decider import each, flow, missing_as, param


def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg


def pipeline():
    return flow(each("items", flow(heavy, name="item"), name="items"), name="order")


FRAME = pl.DataFrame({"items": [[{"weight": 5.0}, {"weight": 25.0}], [], [{"weight": 40.0}]]})


def test_step_into_reaches_the_childs_steps():
    session = pipeline().session(FRAME)
    session.break_at("order/items")
    session.resume()
    cp = session.step_into()
    assert cp.origin.path.startswith("order/items/")
    session.step_into()
    assert session.current.origin.path == "order/items/item/heavy"


def test_value_and_set_work_inside_the_child():
    session = pipeline().session(FRAME)
    session.break_at("order/items")
    session.resume()
    session.step_into()
    session.step_into()  # just before heavy runs on the first item
    assert session.value("weight").to_list() == [5.0]
    session.set("weight", 30.0)
    session.step()
    assert session.value("heavy").to_list() == [True]
    session.resume()
    assert session.output()["items"].to_list()[0][0] == {"weight": 5.0, "heavy": False}
