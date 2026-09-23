"""A debug session driven over a websocket: commands in, events out, values on request."""
import asyncio
import json
import threading

import polars as pl

from decider import flow
from decider.serving.session_ws import session_app


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


pipeline = flow(disposable_income, affordability_ratio)
FRAME = pl.DataFrame({"net_income": [9200.0, 4100.0], "expenses": [3100.0, 1500.0], "instalment": [1200.0, 800.0]})


class Socket:
    # A raw ASGI websocket client: starlette's TestClient needs httpx, which isn't a dependency.
    def __init__(self, app):
        self.inbox, self.outbox = asyncio.Queue(), asyncio.Queue()
        self.inbox.put_nowait({"type": "websocket.connect"})
        scope = {"type": "websocket", "path": "/", "root_path": "", "headers": [], "query_string": b"",
                 "subprotocols": []}
        self.task = asyncio.create_task(app(scope, self.inbox.get, self.outbox.put))

    async def send(self, msg):
        await self.inbox.put({"type": "websocket.receive", "text": msg if isinstance(msg, str) else json.dumps(msg)})

    async def receive(self):
        msg = await asyncio.wait_for(self.outbox.get(), 5)
        if msg["type"] == "websocket.accept":
            return await self.receive()
        return json.loads(msg["text"])

    async def until(self, kind):
        seen = [await self.receive()]
        while seen[-1]["kind"] != kind:
            seen.append(await self.receive())
        return seen

    async def close(self):
        await self.inbox.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self.task, 5)


async def test_break_set_resume_over_the_socket_changes_the_output():
    sessions = []
    ws = Socket(session_app(lambda: sessions.append(pipeline.session(FRAME)) or sessions[-1]))
    assert (await ws.receive())["kind"] == "run_started"
    await ws.send({"kind": "break_at", "target": "affordability_ratio"})
    await ws.send({"kind": "resume"})
    paused = (await ws.until("paused"))[-1]
    assert (paused["origin"]["path"], paused["reason"]) == ("affordability_ratio", "breakpoint")
    await ws.send({"kind": "set", "name": "disposable_income", "value": 1200.0})
    overridden = (await ws.until("overridden"))[-1]
    assert overridden["value"]["preview"] == [1200.0, 1200.0]
    await ws.send({"kind": "resume"})
    finished = (await ws.until("run_finished"))[-1]
    assert finished["output"]["affordability_ratio"]["preview"] == [1.0, 1.5]
    await ws.close()
    assert sessions[0].output()["affordability_ratio"].to_list() == [1.0, 1.5]


async def test_a_malformed_message_is_rejected_and_the_connection_stays_open():
    ws = Socket(session_app(lambda: pipeline.session(FRAME)))
    await ws.receive()
    for bad in ["not json", {"kind": "fly"}, {"kind": "set", "name": "x"}, {"kind": "value"}]:
        await ws.send(bad)
        assert (await ws.until("rejected"))[-1]["message"]
    await ws.send({"kind": "set", "name": "nope", "value": 1.0})
    assert "KeyError" in (await ws.until("rejected"))[-1]["message"]
    await ws.send({"kind": "resume"})
    await ws.until("run_finished")
    await ws.close()


async def test_a_value_request_returns_the_full_column():
    ws = Socket(session_app(lambda: pipeline.session(FRAME)))
    await ws.send({"kind": "resume"})
    await ws.until("run_finished")
    await ws.send({"kind": "value", "spec": "disposable_income"})
    reply = (await ws.until("value"))[-1]
    assert reply == {"kind": "value", "spec": "disposable_income", "dtype": "Float64", "values": [6100.0, 2600.0]}
    await ws.close()


async def test_pause_interrupts_a_running_resume():
    entered, gate = threading.Event(), threading.Event()

    def slow(disposable_income: float) -> float:
        entered.set()
        gate.wait(5)                       # blocks the worker until the pause has arrived
        return disposable_income

    def factory():
        s = flow(disposable_income, slow, affordability_ratio).session(FRAME)
        pause = s.pause
        s.pause = lambda: (pause(), gate.set())
        return s

    ws = Socket(session_app(factory))
    await ws.send({"kind": "resume"})
    assert await asyncio.to_thread(entered.wait, 5)
    await ws.send({"kind": "pause"})
    paused = (await ws.until("paused"))[-1]
    assert (paused["origin"]["path"], paused["when"], paused["reason"]) == ("slow", "after", "pause")
    await ws.send({"kind": "resume"})
    await ws.until("run_finished")
    await ws.close()
