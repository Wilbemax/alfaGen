from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import httpx
import pytest

from scripts.load_check import (
    LoadConfig,
    VirtualClock,
    planned_slot_count,
    run_load,
)

pytestmark = pytest.mark.asyncio


def _config(**overrides: object) -> LoadConfig:
    values: dict[str, object] = {
        "base_url": "http://example.test",
        "rps": 10,
        "duration": 1,
        "concurrency": 4,
        "timeout": 5,
        "strict": True,
    }
    values.update(overrides)
    return LoadConfig(**values)  # type: ignore[arg-type]


def _roundtrip(clock: VirtualClock | None = None, delay: float = 0.0) -> tuple[httpx.MockTransport, dict[str, int]]:
    originals: dict[str, str] = {}
    calls: dict[str, int] = defaultdict(int)

    async def handler(request: httpx.Request) -> httpx.Response:
        if delay and clock is not None:
            await clock.sleep(delay)
        body = json.loads(request.content.decode())
        payload_id = body["payload_id"]
        payload = body["payload"]
        calls[payload_id] += 1
        if payload_id not in originals:
            originals[payload_id] = payload
            return httpx.Response(200, json={"result": "*" * len(payload)})
        if payload == "*" * len(originals[payload_id]):
            return httpx.Response(200, json={"result": originals[payload_id]})
        return httpx.Response(200, json={"result": "WRONG"})

    return httpx.MockTransport(handler), calls


async def test_more_than_one_request_is_in_flight() -> None:
    clock = VirtualClock()
    entered = {"current": 0, "peak": 0}
    release_at = 2

    class Gate:
        def __init__(self) -> None:
            self._waiters: list[asyncio.Future[None]] = []
            self._open = False

        async def wait(self) -> None:
            if self._open:
                return
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(future)
            await future

        def open(self) -> None:
            self._open = True
            for future in self._waiters:
                if not future.done():
                    future.set_result(None)

    gate = Gate()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        entered["current"] += 1
        entered["peak"] = max(entered["peak"], entered["current"])
        if entered["current"] >= release_at:
            gate.open()
        await gate.wait()
        entered["current"] -= 1
        payload = body["payload"]
        return httpx.Response(200, json={"result": payload})

    report = await run_load(
        _config(rps=20, duration=0.5, concurrency=8),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert entered["peak"] >= 2
    assert report.peak_in_flight >= 2


async def test_scheduler_creates_expected_slots() -> None:
    assert planned_slot_count(30, 1000) == 30000
    assert planned_slot_count(1, 10) == 10
    clock = VirtualClock()
    transport, calls = _roundtrip(clock)
    report = await run_load(_config(rps=10, duration=1, concurrency=4), transport=transport, clock=clock)
    assert report.scheduled == 10
    assert len(calls) == 10
    assert len(set(calls)) == 10


async def test_semaphore_limits_concurrency() -> None:
    clock = VirtualClock()
    current = {"value": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        current["value"] += 1
        current["peak"] = max(current["peak"], current["value"])
        await clock.sleep(0.05)
        current["value"] -= 1
        body = json.loads(request.content.decode())
        return httpx.Response(200, json={"result": body["payload"]})

    report = await run_load(
        _config(rps=20, duration=0.5, concurrency=1),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert current["peak"] == 1
    assert report.peak_in_flight == 1
    assert report.scheduled == 10


async def test_retry_after_and_at_most_two_retries() -> None:
    clock = VirtualClock()
    attempts: dict[str, list[float]] = defaultdict(list)
    originals: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        payload_id = body["payload_id"]
        attempts[payload_id].append(clock.monotonic())
        if payload_id not in originals:
            originals[payload_id] = body["payload"]
        if len(attempts[payload_id]) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.4"})
        if body["payload"] == originals[payload_id]:
            return httpx.Response(200, json={"result": "*" * len(body["payload"])})
        return httpx.Response(200, json={"result": originals[payload_id]})

    report = await run_load(
        _config(rps=2, duration=1, concurrency=1),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.scheduled == 2
    assert report.retried_requests >= 2
    assert report.final_429 == 0
    for stamps in attempts.values():
        assert len(stamps) >= 2
        assert stamps[1] - stamps[0] >= 0.4 - 1e-9
        assert len(stamps) <= 3


async def test_max_two_retries_on_server_error() -> None:
    clock = VirtualClock()
    calls = {"count": 0}

    async def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500)

    report = await run_load(
        _config(rps=1, duration=1, concurrency=1),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.scheduled == 1
    assert calls["count"] == 3
    assert report.final_5xx == 1


async def test_five_invalid_responses_stop_the_run() -> None:
    clock = VirtualClock()
    calls = {"count": 0}

    async def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500)

    report = await run_load(
        _config(rps=10, duration=2, concurrency=1, strict=False),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.scheduled == 5
    assert calls["count"] == 15
    assert report.max_consecutive_invalid == 5
    assert report.passed is False
    assert any("consecutive invalid" in reason for reason in report.reasons)


async def test_429_does_not_reset_invalid_counter() -> None:
    clock = VirtualClock()
    order: list[str] = []
    attempts: dict[str, int] = defaultdict(int)
    script = (
        [500, 500, 500],
        [429, 429, 429],
        [500, 500, 500],
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        payload_id = body["payload_id"]
        if payload_id not in order:
            order.append(payload_id)
        index = order.index(payload_id)
        attempt = attempts[payload_id]
        attempts[payload_id] += 1
        status = script[index][attempt]
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(status)

    report = await run_load(
        _config(rps=3, duration=1, concurrency=1, strict=False),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.max_consecutive_invalid == 2
    assert report.consecutive_invalid == 2
    assert report.final_429 == 1


async def test_success_resets_invalid_counter() -> None:
    clock = VirtualClock()
    order: list[str] = []
    attempts: dict[str, int] = defaultdict(int)
    originals: dict[str, str] = {}
    script = (
        [500, 500, 500],
        [200],
        [500, 500, 500],
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        payload_id = body["payload_id"]
        payload = body["payload"]
        if payload_id not in originals:
            originals[payload_id] = payload
        if payload != originals[payload_id]:
            return httpx.Response(200, json={"result": originals[payload_id]})
        if payload_id not in order:
            order.append(payload_id)
        index = order.index(payload_id)
        attempt = attempts[payload_id]
        attempts[payload_id] += 1
        status = script[index][attempt]
        if status == 200:
            return httpx.Response(200, json={"result": "*" * len(payload)})
        return httpx.Response(status)

    report = await run_load(
        _config(rps=3, duration=1, concurrency=1, strict=False),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.max_consecutive_invalid == 1
    assert report.success == 1


async def test_demask_mismatch_fails() -> None:
    clock = VirtualClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body["payload"].startswith("*"):
            return httpx.Response(200, json={"result": "NOT-THE-SOURCE"})
        return httpx.Response(200, json={"result": "*" * len(body["payload"])})

    report = await run_load(
        _config(rps=2, duration=1, concurrency=2),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.demask_mismatches > 0
    assert report.passed is False
    assert any("demask mismatches" in reason for reason in report.reasons)


async def test_low_achieved_rps_fails_strict_gate() -> None:
    clock = VirtualClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        await clock.sleep(0.5)
        body = json.loads(request.content.decode())
        return httpx.Response(200, json={"result": body["payload"]})

    report = await run_load(
        _config(rps=20, duration=1, concurrency=1),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.completion_rps < 0.95 * report.target_rps
    assert report.passed is False
    assert any("completion RPS" in reason for reason in report.reasons)


async def test_high_p95_fails_strict_gate() -> None:
    clock = VirtualClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        await clock.sleep(1.2)
        body = json.loads(request.content.decode())
        return httpx.Response(200, json={"result": body["payload"]})

    report = await run_load(
        _config(rps=1, duration=10, concurrency=10),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.mask_p95 > 1.0
    assert report.passed is False
    assert any("p95" in reason for reason in report.reasons)


async def test_correct_run_passes_strict_gate() -> None:
    clock = VirtualClock()
    transport, calls = _roundtrip(clock)
    report = await run_load(_config(rps=10, duration=1, concurrency=4), transport=transport, clock=clock)
    assert report.scheduled == 10
    assert report.success == 10
    assert report.demask_mismatches == 0
    assert report.final_5xx == 0
    assert report.final_429 == 0
    assert report.mask_p95 <= 1.0
    assert report.demask_p95 <= 1.0
    assert report.completion_rps >= 0.95 * report.target_rps
    assert report.max_consecutive_invalid < 5
    assert report.passed is True
    assert len(calls) == 10
