from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from itertools import pairwise
from types import SimpleNamespace

import httpx
import pytest

from scripts.load_check import (
    LoadConfig,
    LoadReport,
    Outcome,
    VirtualClock,
    _record_result,
    _RoundtripResult,
    planned_slot_count,
    run_from_cli,
    run_load,
)

pytestmark = pytest.mark.asyncio


def _config(**overrides: object) -> LoadConfig:
    values: dict[str, object] = {
        "base_url": "http://example.test",
        "rps": 10,
        "duration": 1,
        "concurrency": 20,
        "timeout": 5,
        "drain_timeout": 5,
        "strict": True,
    }
    values.update(overrides)
    return LoadConfig(**values)  # type: ignore[arg-type]


def _roundtrip(
    clock: VirtualClock,
    *,
    mask_delay: float = 0.0,
    demask_delay: float = 0.0,
) -> tuple[httpx.MockTransport, list[tuple[str, str, float]]]:
    originals: dict[str, str] = {}
    events: list[tuple[str, str, float]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        payload_id, payload = body["payload_id"], body["payload"]
        if payload_id not in originals:
            originals[payload_id] = payload
            events.append(("mask", payload_id, clock.monotonic()))
            await clock.sleep(mask_delay)
            return httpx.Response(200, json={"result": "*" * len(payload)})
        events.append(("demask", payload_id, clock.monotonic()))
        await clock.sleep(demask_delay)
        result = originals[payload_id] if payload == "*" * len(originals[payload_id]) else "WRONG"
        return httpx.Response(200, json={"result": result})

    return httpx.MockTransport(handler), events


async def test_scheduler_creates_expected_roundtrip_slots() -> None:
    assert planned_slot_count(30, 1000) == 30000
    clock = VirtualClock()
    transport, _events = _roundtrip(clock)
    report = await run_load(_config(rps=10), transport=transport, clock=clock)
    assert report.scheduled_roundtrips == 10
    assert report.started_roundtrips == 10
    assert report.completed_roundtrips == 10
    assert report.successful_roundtrips == 10


async def test_roundtrips_are_evenly_paced_without_catch_up_burst() -> None:
    clock = VirtualClock()
    transport, events = _roundtrip(clock)
    report = await run_load(_config(rps=5, duration=1), transport=transport, clock=clock)
    mask_starts = [stamp for phase, _pid, stamp in events if phase == "mask"]
    assert len(mask_starts) == 5
    assert all(b - a >= 0.2 - 1e-9 for a, b in pairwise(mask_starts))
    assert report.pacing_drops == 0


async def test_demask_starts_before_global_mask_completion() -> None:
    clock = VirtualClock()
    transport, events = _roundtrip(clock, mask_delay=0.01, demask_delay=0.01)
    report = await run_load(_config(rps=10), transport=transport, clock=clock)
    phases = [phase for phase, _pid, _stamp in events]
    assert phases.index("demask") < max(i for i, phase in enumerate(phases) if phase == "mask")
    assert report.demask_logical_completed == report.mask_success == 10


async def test_roundtrip_and_http_rps_are_counted_separately() -> None:
    clock = VirtualClock()
    transport, _events = _roundtrip(clock)
    report = await run_load(_config(rps=4, duration=2), transport=transport, clock=clock)
    assert report.actual_success_rps == pytest.approx(4.0)
    assert report.mask_http_rps == pytest.approx(4.0)
    assert report.demask_http_rps == pytest.approx(4.0)
    assert report.total_http_rps == pytest.approx(8.0)
    assert report.http_attempts == 16


async def test_completed_result_is_aggregated_without_a_result_buffer() -> None:
    report = LoadReport(target_rps=1, duration=1, strict=True)
    mask_latencies: list[float] = []
    demask_latencies: list[float] = []
    roundtrip_latencies: list[float] = []
    _record_result(
        report,
        _RoundtripResult(
            mask=Outcome("success", 200, "***"),
            mask_latency=0.01,
            demask=Outcome("success", 200, "source"),
            demask_latency=0.02,
            roundtrip_latency=0.03,
        ),
        mask_latencies,
        demask_latencies,
        roundtrip_latencies,
    )
    assert report.completed_roundtrips == report.successful_roundtrips == 1
    assert report.mask_http_attempts == report.demask_http_attempts == 1
    assert mask_latencies == [0.01]
    assert demask_latencies == [0.02]
    assert roundtrip_latencies == [0.03]


async def test_cli_uses_uvloop_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = LoadReport(target_rps=1, duration=1, strict=False)

    def fake_run(coroutine: object) -> LoadReport:
        coroutine.close()  # type: ignore[attr-defined]
        return expected

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "uvloop", SimpleNamespace(run=fake_run))
    assert run_from_cli(_config()) is expected


async def test_mask_demask_and_full_roundtrip_latency_are_separate() -> None:
    clock = VirtualClock()
    transport, _events = _roundtrip(clock, mask_delay=0.1, demask_delay=0.2)
    report = await run_load(
        _config(rps=1, duration=2, concurrency=2), transport=transport, clock=clock
    )
    assert report.mask_p50 == pytest.approx(0.1)
    assert report.demask_p50 == pytest.approx(0.2)
    assert report.roundtrip_p50 == pytest.approx(0.3)
    assert report.roundtrip_max >= report.mask_max + report.demask_max - 1e-9


async def test_concurrency_saturation_is_bounded_and_fails() -> None:
    clock = VirtualClock()
    transport, _events = _roundtrip(clock, mask_delay=0.6, demask_delay=0.6)
    report = await run_load(
        _config(rps=10, duration=1, concurrency=1), transport=transport, clock=clock
    )
    assert report.peak_in_flight == 1
    assert report.saturation_drops > 0
    assert report.started_roundtrips < report.scheduled_roundtrips
    assert report.passed is False
    assert any("saturation" in reason for reason in report.reasons)


async def test_demask_mismatch_is_a_correctness_failure() -> None:
    clock = VirtualClock()
    originals: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        pid, payload = body["payload_id"], body["payload"]
        if pid not in originals:
            originals.add(pid)
            return httpx.Response(200, json={"result": "*" * len(payload)})
        return httpx.Response(200, json={"result": "NOT-THE-SOURCE"})

    report = await run_load(
        _config(rps=2), transport=httpx.MockTransport(handler), clock=clock
    )
    assert report.demask_mismatches == 2
    assert report.successful_roundtrips == 0
    assert any("demask mismatches" in reason for reason in report.reasons)


@pytest.mark.parametrize(
    ("status", "field"),
    [(429, "final_429"), (500, "final_5xx")],
)
async def test_final_http_errors_are_counted(status: int, field: str) -> None:
    clock = VirtualClock()

    async def handler(_request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": "0"} if status == 429 else None
        return httpx.Response(status, headers=headers)

    report = await run_load(
        _config(rps=1), transport=httpx.MockTransport(handler), clock=clock
    )
    assert getattr(report, field) == 1
    assert report.mask_http_attempts == 3
    assert report.passed is False


async def test_network_error_is_counted() -> None:
    clock = VirtualClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unavailable", request=request)

    report = await run_load(
        _config(rps=1), transport=httpx.MockTransport(handler), clock=clock
    )
    assert report.network_errors == 1
    assert report.timeouts == 0


async def test_timeout_is_counted_separately() -> None:
    clock = VirtualClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    report = await run_load(
        _config(rps=1), transport=httpx.MockTransport(handler), clock=clock
    )
    assert report.timeouts == 1
    assert report.network_errors == 0


async def test_retry_success_counts_attempts_without_final_error() -> None:
    clock = VirtualClock()
    attempts: dict[str, int] = defaultdict(int)
    originals: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        pid, payload = body["payload_id"], body["payload"]
        originals.setdefault(pid, payload)
        attempts[pid] += 1
        if attempts[pid] == 1:
            return httpx.Response(500)
        if payload == originals[pid]:
            return httpx.Response(200, json={"result": "*" * len(payload)})
        return httpx.Response(200, json={"result": originals[pid]})

    report = await run_load(
        _config(rps=1), transport=httpx.MockTransport(handler), clock=clock
    )
    assert report.mask_http_attempts == 2
    assert report.demask_http_attempts == 1
    assert report.final_5xx == 0
    assert report.successful_roundtrips == 1


async def test_strict_pass() -> None:
    clock = VirtualClock()
    transport, _events = _roundtrip(clock, mask_delay=0.01, demask_delay=0.01)
    report = await run_load(_config(rps=10), transport=transport, clock=clock)
    assert report.actual_success_rps >= 0.95 * report.target_roundtrip_rps
    assert report.roundtrip_p95 <= 1.0
    assert report.passed is True


async def test_strict_fails_when_actual_success_is_below_95_percent() -> None:
    clock = VirtualClock()
    transport, _events = _roundtrip(clock, mask_delay=0.5, demask_delay=0.5)
    report = await run_load(
        _config(rps=20, concurrency=1), transport=transport, clock=clock
    )
    assert report.actual_success_rps < 0.95 * report.target_roundtrip_rps
    assert any("95%" in reason for reason in report.reasons)


async def test_roundtrip_latency_gate_is_explicit() -> None:
    clock = VirtualClock()
    transport, _events = _roundtrip(clock, mask_delay=0.6, demask_delay=0.6)
    report = await run_load(
        _config(rps=1, duration=2, concurrency=2), transport=transport, clock=clock
    )
    assert report.mask_p95 <= 1.0
    assert report.demask_p95 <= 1.0
    assert report.roundtrip_p95 > 1.0
    assert any("roundtrip p95" in reason for reason in report.reasons)


async def test_drain_timeout_cancels_runaway_backlog() -> None:
    clock = VirtualClock()

    async def handler(_request: httpx.Request) -> httpx.Response:
        # A request that never resolves lets the runner's drain deadline win
        # without introducing a competing long virtual-clock sleep.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    report = await run_load(
        _config(rps=1, duration=1, drain_timeout=0.5),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert report.drain_timed_out is True
    assert any("drain timeout" in reason for reason in report.reasons)


async def test_five_consecutive_5xx_stop_future_scheduling() -> None:
    clock = VirtualClock()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    report = await run_load(
        _config(rps=10, duration=2, concurrency=1, strict=False),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    # One already-admitted chain may finish concurrently with the fifth
    # failure before the scheduler observes the stop flag.
    assert report.max_consecutive_5xx >= 5
    assert report.started_roundtrips < report.scheduled_roundtrips
    assert any("consecutive invalid" in reason for reason in report.reasons)
