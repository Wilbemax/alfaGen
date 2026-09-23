#!/usr/bin/env python3
"""Steady-state open-loop roundtrip load check for POST /process.

Each paced logical operation is one complete chain::

    mask(original, payload_id) -> demask(masked, same payload_id)

New chains are admitted independently of earlier completions. The pacer never
bursts to catch up after falling behind, and concurrency bounds the backlog.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import heapq
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import httpx

SYNTHETIC_PAYLOADS = (
    "Заявитель Орлов Кирилл Денисович, паспорт 3916 204418",
    "Телефон для связи +7 900 555-01-02 и почта olga.orlova@example.net",
    "Дата рождения 03.11.1985, ИНН 500100732259",
)
MAX_RETRIES = 2
INVALID_STREAK_LIMIT = 5
LATENCY_GATE_SECONDS = 1.0


class Clock(Protocol):
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(seconds, 0.0))


class VirtualClock:
    """Virtual clock used by unit tests."""

    def __init__(self) -> None:
        self._time = 0.0
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0
        self._wake = asyncio.Event()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def monotonic(self) -> float:
        return self._time

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        self._closed = True
        self._wake.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self._time + seconds, self._seq, future))
        self._seq += 1
        self._wake.set()
        await future

    async def _pump(self) -> None:
        while not self._closed:
            await self._wake.wait()
            if self._closed:
                return
            self._wake.clear()
            await asyncio.sleep(0)
            self._advance_once()
            if self._waiters and not self._closed:
                self._wake.set()

    def _advance_once(self) -> None:
        pending = [item for item in self._waiters if not item[2].done()]
        heapq.heapify(pending)
        self._waiters = pending
        if not self._waiters:
            return
        self._time = max(self._time, self._waiters[0][0])
        while self._waiters and self._waiters[0][0] <= self._time + 1e-9:
            _wake_at, _seq, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(None)


@dataclass(slots=True)
class LoadConfig:
    base_url: str = "http://127.0.0.1:8000"
    rps: float = 1000.0  # target roundtrip RPS
    duration: float = 30.0
    concurrency: int = 200
    timeout: float = 10.0
    drain_timeout: float = 30.0
    max_connections: int | None = None
    max_keepalive_connections: int | None = None
    strict: bool = False


@dataclass(slots=True)
class Outcome:
    kind: str
    status: int
    body: str | None = None
    attempts: int = 1


@dataclass(slots=True)
class _RoundtripResult:
    mask: Outcome
    mask_latency: float
    demask: Outcome | None
    demask_latency: float | None
    roundtrip_latency: float | None
    mismatch: bool = False


@dataclass(slots=True)
class LoadReport:
    target_rps: float
    duration: float
    strict: bool
    scheduled_roundtrips: int = 0
    started_roundtrips: int = 0
    completed_roundtrips: int = 0
    successful_roundtrips: int = 0
    actual_started_rps: float = 0.0
    actual_completed_rps: float = 0.0
    actual_success_rps: float = 0.0
    saturation_drops: int = 0
    pacing_drops: int = 0
    drain_timed_out: bool = False
    mask_logical_scheduled: int = 0
    mask_logical_started: int = 0
    mask_logical_completed: int = 0
    mask_http_attempts: int = 0
    mask_success: int = 0
    mask_final_429: int = 0
    mask_final_5xx: int = 0
    mask_network_errors: int = 0
    mask_timeouts: int = 0
    mask_contract_errors: int = 0
    mask_http_rps: float = 0.0
    mask_achieved_completion_rps: float = 0.0
    mask_p50: float = 0.0
    mask_p95: float = 0.0
    mask_p99: float = 0.0
    mask_max: float = 0.0
    demask_logical_scheduled: int = 0
    demask_logical_started: int = 0
    demask_logical_completed: int = 0
    demask_http_attempts: int = 0
    demask_success: int = 0
    demask_final_429: int = 0
    demask_final_5xx: int = 0
    demask_network_errors: int = 0
    demask_timeouts: int = 0
    demask_contract_errors: int = 0
    demask_http_rps: float = 0.0
    demask_p50: float = 0.0
    demask_p95: float = 0.0
    demask_p99: float = 0.0
    demask_max: float = 0.0
    demask_mismatches: int = 0
    roundtrip_p50: float = 0.0
    roundtrip_p95: float = 0.0
    roundtrip_p99: float = 0.0
    roundtrip_max: float = 0.0
    total_http_rps: float = 0.0
    http_attempts: int = 0
    success: int = 0
    final_429: int = 0
    final_5xx: int = 0
    network_errors: int = 0
    timeouts: int = 0
    contract_errors: int = 0
    max_consecutive_5xx: int = 0
    max_consecutive_invalid: int = 0
    consecutive_invalid: int = 0
    retried_requests: int = 0
    final_4xx: int = 0
    elapsed_seconds: float = 0.0
    peak_in_flight: int = 0
    # Compatibility aliases; all now refer to complete roundtrips.
    scheduled: int = 0
    started: int = 0
    completed: int = 0
    scheduling_rps: float = 0.0
    completion_rps: float = 0.0
    fraction_latency_gt_1s: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def target_roundtrip_rps(self) -> float:
        return self.target_rps

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(slots=True)
class _RunState:
    consecutive_invalid: int = 0
    max_consecutive_invalid: int = 0
    consecutive_5xx: int = 0
    max_consecutive_5xx: int = 0
    stop: bool = False
    in_flight: int = 0
    peak_in_flight: int = 0


def planned_slot_count(duration: float, rps: float) -> int:
    if duration <= 0 or rps <= 0:
        return 0
    return math.floor(duration * rps + 1e-9)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def failure_reasons(report: LoadReport, *, strict: bool) -> list[str]:
    reasons: list[str] = []
    checks = (
        (report.demask_mismatches, f"demask mismatches: {report.demask_mismatches}"),
        (report.final_5xx, f"final 5xx: {report.final_5xx}"),
        (report.final_429, f"final 429: {report.final_429}"),
        (report.final_4xx, f"final 4xx: {report.final_4xx}"),
        (report.network_errors, f"network errors: {report.network_errors}"),
        (report.timeouts, f"timeouts: {report.timeouts}"),
        (report.contract_errors, f"invalid JSON contract: {report.contract_errors}"),
        (report.saturation_drops, f"saturation drops: {report.saturation_drops}"),
    )
    reasons.extend(message for value, message in checks if value)
    if report.max_consecutive_invalid >= INVALID_STREAK_LIMIT:
        reasons.append(f"max consecutive invalid >= {INVALID_STREAK_LIMIT}")
    if report.drain_timed_out:
        reasons.append("drain timeout exceeded")
    if strict:
        if report.actual_success_rps < 0.95 * report.target_roundtrip_rps:
            reasons.append(
                f"actual success RPS {report.actual_success_rps:.2f} "
                f"< 95% of target roundtrip RPS {report.target_roundtrip_rps:.2f}"
            )
        if report.mask_p95 > LATENCY_GATE_SECONDS:
            reasons.append(f"mask p95 {report.mask_p95:.4f}s > 1s")
        if report.demask_p95 > LATENCY_GATE_SECONDS:
            reasons.append(f"demask p95 {report.demask_p95:.4f}s > 1s")
        if report.roundtrip_p95 > LATENCY_GATE_SECONDS:
            reasons.append(f"roundtrip p95 {report.roundtrip_p95:.4f}s > 1s")
    return reasons


def _note_final(state: _RunState, kind: str) -> None:
    if kind == "success":
        state.consecutive_invalid = 0
        state.consecutive_5xx = 0
        return
    if kind == "rate_limited":
        return
    state.consecutive_invalid += 1
    state.max_consecutive_invalid = max(state.max_consecutive_invalid, state.consecutive_invalid)
    if kind == "server_error":
        state.consecutive_5xx += 1
        state.max_consecutive_5xx = max(state.max_consecutive_5xx, state.consecutive_5xx)
    if state.consecutive_invalid >= INVALID_STREAK_LIMIT:
        state.stop = True


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", "0")))
    except ValueError:
        return 0.0


def _classify(response: httpx.Response) -> Outcome:
    status = response.status_code
    if status == 429:
        return Outcome("rate_limited", 429)
    if status == 200:
        try:
            body = response.json()
        except json.JSONDecodeError:
            return Outcome("invalid", 200)
        if not isinstance(body, dict) or not isinstance(body.get("result"), str):
            return Outcome("invalid", 200)
        return Outcome("success", 200, body["result"])
    if 400 <= status < 500:
        return Outcome("invalid", status)
    if status >= 500:
        return Outcome("server_error", status)
    return Outcome("invalid", status)


async def _request_with_retries(
    client: httpx.AsyncClient, clock: Clock, state: _RunState,
    payload: str, payload_id: str, timeout: float,
) -> tuple[Outcome, float]:
    started = clock.monotonic()
    last = Outcome("invalid", 0)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = await client.post(
                "/process", json={"payload": payload, "payload_id": payload_id}, timeout=timeout
            )
        except httpx.TimeoutException:
            last = Outcome("timeout", 0, attempts=attempt)
        except httpx.HTTPError:
            last = Outcome("network_error", 0, attempts=attempt)
        else:
            last = _classify(response)
            last.attempts = attempt
            if last.kind == "success" or attempt > MAX_RETRIES:
                break
            if last.kind == "rate_limited":
                await clock.sleep(_retry_after_seconds(response))
                continue
        if attempt > MAX_RETRIES:
            break
        await clock.sleep(0)
    _note_final(state, last.kind)
    return last, max(clock.monotonic() - started, 0.0)


def _count_failure(report: LoadReport, outcome: Outcome, *, phase: str) -> None:
    prefix = "mask" if phase == "mask" else "demask"
    field_name: str | None = None
    if outcome.kind == "rate_limited":
        field_name = f"{prefix}_final_429"
    elif outcome.kind == "timeout":
        field_name = f"{prefix}_timeouts"
    elif outcome.kind == "network_error":
        field_name = f"{prefix}_network_errors"
    elif outcome.status == 200:
        field_name = f"{prefix}_contract_errors"
    elif 400 <= outcome.status < 500:
        report.final_4xx += 1
    elif outcome.status >= 500:
        field_name = f"{prefix}_final_5xx"
    if field_name:
        setattr(report, field_name, getattr(report, field_name) + 1)


async def run_load(
    config: LoadConfig, *, transport: httpx.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> LoadReport:
    clock = clock or SystemClock()
    virtual = clock if isinstance(clock, VirtualClock) else None
    if virtual is not None:
        virtual.start()
    report = LoadReport(config.rps, config.duration, config.strict)
    try:
        await _execute(config, report, transport=transport, clock=clock)
    finally:
        if virtual is not None:
            await virtual.stop()
    report.reasons = failure_reasons(report, strict=config.strict)
    return report


async def _execute(
    config: LoadConfig, report: LoadReport, *,
    transport: httpx.AsyncBaseTransport | None, clock: Clock,
) -> None:
    max_connections = config.max_connections or max(config.concurrency, 1)
    keepalive = min(config.max_keepalive_connections or max_connections, max_connections)
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=keepalive)
    semaphore = asyncio.Semaphore(max(config.concurrency, 1))
    state = _RunState()
    mask_latencies: list[float] = []
    demask_latencies: list[float] = []
    roundtrip_latencies: list[float] = []
    results: list[_RoundtripResult] = []
    tasks: set[asyncio.Task[_RoundtripResult]] = set()

    async with httpx.AsyncClient(
        base_url=config.base_url, limits=limits, timeout=config.timeout, transport=transport
    ) as client:
        run_started = clock.monotonic()
        schedule_end = run_started + max(config.duration, 0.0)
        interval = 1.0 / config.rps if config.rps > 0 else math.inf
        report.scheduled_roundtrips = planned_slot_count(config.duration, config.rps)
        overdue_admission_used = False

        async def one_roundtrip(index: int) -> _RoundtripResult:
            state.in_flight += 1
            state.peak_in_flight = max(state.peak_in_flight, state.in_flight)
            chain_started = clock.monotonic()
            payload = SYNTHETIC_PAYLOADS[index % len(SYNTHETIC_PAYLOADS)]
            payload_id = uuid.uuid4().hex
            try:
                mask, mask_latency = await _request_with_retries(
                    client, clock, state, payload, payload_id, config.timeout
                )
                if mask.kind != "success" or mask.body is None:
                    return _RoundtripResult(mask, mask_latency, None, None, None)
                demask, demask_latency = await _request_with_retries(
                    client, clock, state, mask.body, payload_id, config.timeout
                )
                return _RoundtripResult(
                    mask, mask_latency, demask, demask_latency,
                    max(clock.monotonic() - chain_started, 0.0),
                    demask.kind == "success" and demask.body != payload,
                )
            finally:
                state.in_flight -= 1
                semaphore.release()

        def collect(task: asyncio.Task[_RoundtripResult]) -> None:
            tasks.discard(task)
            if not task.cancelled() and task.exception() is None:
                results.append(task.result())

        for index in range(report.scheduled_roundtrips):
            if state.stop:
                report.pacing_drops += report.scheduled_roundtrips - index
                break
            planned_at = run_started + index * interval
            delay = planned_at - clock.monotonic()
            if delay > 0:
                await clock.sleep(delay)
                overdue_admission_used = False
            elif overdue_admission_used:
                report.pacing_drops += 1
                continue
            else:
                overdue_admission_used = True
            now = clock.monotonic()
            if now >= schedule_end:
                report.pacing_drops += report.scheduled_roundtrips - index
                break
            if semaphore.locked():
                report.saturation_drops += 1
                continue
            await semaphore.acquire()
            report.started_roundtrips += 1
            task = asyncio.create_task(one_roundtrip(index))
            tasks.add(task)
            task.add_done_callback(collect)

        await clock.sleep(schedule_end - clock.monotonic())
        report.actual_started_rps = report.started_roundtrips / max(config.duration, 1e-9)

        if tasks:
            drain = asyncio.create_task(clock.sleep(max(config.drain_timeout, 0.0)))
            gathered = asyncio.gather(*tuple(tasks), return_exceptions=True)
            done, _pending = await asyncio.wait({gathered, drain}, return_when=asyncio.FIRST_COMPLETED)
            if gathered not in done:
                report.drain_timed_out = True
                for task in tuple(tasks):
                    task.cancel()
                await asyncio.gather(*tuple(tasks), return_exceptions=True)
            else:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain
        report.elapsed_seconds = max(clock.monotonic() - run_started, config.duration, 1e-9)

    for result in results:
        report.completed_roundtrips += 1
        report.mask_logical_completed += 1
        report.mask_http_attempts += result.mask.attempts
        mask_latencies.append(result.mask_latency)
        report.retried_requests += int(result.mask.attempts > 1)
        if result.mask.kind == "success":
            report.mask_success += 1
        else:
            _count_failure(report, result.mask, phase="mask")
        if result.demask is not None and result.demask_latency is not None:
            report.demask_logical_scheduled += 1
            report.demask_logical_started += 1
            report.demask_logical_completed += 1
            report.demask_http_attempts += result.demask.attempts
            demask_latencies.append(result.demask_latency)
            if result.roundtrip_latency is not None:
                roundtrip_latencies.append(result.roundtrip_latency)
            report.retried_requests += int(result.demask.attempts > 1)
            if result.demask.kind == "success":
                report.demask_success += 1
                if result.mismatch:
                    report.demask_mismatches += 1
                else:
                    report.successful_roundtrips += 1
            else:
                _count_failure(report, result.demask, phase="demask")

    report.mask_logical_scheduled = report.scheduled_roundtrips
    report.mask_logical_started = report.started_roundtrips
    report.mask_achieved_completion_rps = report.mask_logical_completed / report.elapsed_seconds
    report.actual_completed_rps = report.completed_roundtrips / report.elapsed_seconds
    report.actual_success_rps = report.successful_roundtrips / report.elapsed_seconds
    report.mask_http_rps = report.mask_http_attempts / report.elapsed_seconds
    report.demask_http_rps = report.demask_http_attempts / report.elapsed_seconds
    report.total_http_rps = (report.mask_http_attempts + report.demask_http_attempts) / report.elapsed_seconds
    for prefix, values in (("mask", mask_latencies), ("demask", demask_latencies), ("roundtrip", roundtrip_latencies)):
        setattr(report, f"{prefix}_p50", percentile(values, 50))
        setattr(report, f"{prefix}_p95", percentile(values, 95))
        setattr(report, f"{prefix}_p99", percentile(values, 99))
        setattr(report, f"{prefix}_max", max(values, default=0.0))
    report.final_429 = report.mask_final_429 + report.demask_final_429
    report.final_5xx = report.mask_final_5xx + report.demask_final_5xx
    report.network_errors = report.mask_network_errors + report.demask_network_errors
    report.timeouts = report.mask_timeouts + report.demask_timeouts
    report.contract_errors = report.mask_contract_errors + report.demask_contract_errors
    report.http_attempts = report.mask_http_attempts + report.demask_http_attempts
    report.max_consecutive_invalid = state.max_consecutive_invalid
    report.consecutive_invalid = state.consecutive_invalid
    report.max_consecutive_5xx = state.max_consecutive_5xx
    report.peak_in_flight = state.peak_in_flight
    report.scheduled = report.scheduled_roundtrips
    report.started = report.started_roundtrips
    report.completed = report.completed_roundtrips
    report.scheduling_rps = report.actual_started_rps
    report.completion_rps = report.actual_completed_rps
    report.success = report.successful_roundtrips
    if roundtrip_latencies:
        report.fraction_latency_gt_1s = sum(x > 1.0 for x in roundtrip_latencies) / len(roundtrip_latencies)


def print_report(report: LoadReport) -> None:
    print(f"target_roundtrip_rps: {report.target_roundtrip_rps}")
    print(f"duration: {report.duration}")
    print(f"strict: {str(report.strict).lower()}")
    print("roundtrips:")
    for name in ("scheduled_roundtrips", "started_roundtrips", "completed_roundtrips", "successful_roundtrips"):
        print(f"  {name.removesuffix('_roundtrips')}: {getattr(report, name)}")
    print(f"  actual_started_rps: {report.actual_started_rps:.2f}")
    print(f"  actual_completed_rps: {report.actual_completed_rps:.2f}")
    print(f"  actual_success_rps: {report.actual_success_rps:.2f}")
    print(f"  saturation_drops: {report.saturation_drops}")
    print(f"  pacing_drops: {report.pacing_drops}")
    print(f"  drain_timed_out: {str(report.drain_timed_out).lower()}")
    print(f"  latency_p50: {report.roundtrip_p50:.4f}")
    print(f"  latency_p95: {report.roundtrip_p95:.4f}")
    print(f"  latency_p99: {report.roundtrip_p99:.4f}")
    print(f"  latency_max: {report.roundtrip_max:.4f}")
    for phase in ("mask", "demask"):
        print(f"{phase}:")
        print(f"  http_attempts: {getattr(report, phase + '_http_attempts')}")
        print(f"  http_rps: {getattr(report, phase + '_http_rps'):.2f}")
        print(f"  success: {getattr(report, phase + '_success')}")
        for pct in ("p50", "p95", "p99", "max"):
            print(f"  latency_{pct}: {getattr(report, phase + '_' + pct):.4f}")
    print(f"  mismatches: {report.demask_mismatches}")
    print("totals:")
    print(f"  http_attempts: {report.http_attempts}")
    print(f"  total_http_rps: {report.total_http_rps:.2f}")
    print(f"  final_429: {report.final_429}")
    print(f"  final_5xx: {report.final_5xx}")
    print(f"  network_errors: {report.network_errors}")
    print(f"  timeouts: {report.timeouts}")
    print(f"  contract_errors: {report.contract_errors}")
    print(f"  retried_requests: {report.retried_requests}")
    print(f"  peak_in_flight_roundtrips: {report.peak_in_flight}")
    print(f"  elapsed_seconds: {report.elapsed_seconds:.4f}")
    for reason in report.reasons:
        print(f"fail_reason: {reason}")
    print(f"result: {'PASS' if report.passed else 'FAIL'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steady-state roundtrip load check for PII gateway")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rps", type=float, default=1000.0, help="Target roundtrip chains per second")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--max-keepalive-connections", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Apply throughput, correctness and latency gates")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LoadConfig:
    return LoadConfig(
        base_url=args.base_url, rps=args.rps, duration=args.duration,
        concurrency=args.concurrency, timeout=args.timeout, drain_timeout=args.drain_timeout,
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections, strict=args.strict,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timer_period_enabled = False
    if sys.platform == "win32":
        # The default Windows timer quantum (~15.6 ms) caps an evenly paced
        # open-loop generator near 128 starts/s. A 1 ms period is scoped to the
        # CLI run and restored in finally; pacing/drop semantics stay unchanged.
        timer_period_enabled = ctypes.windll.winmm.timeBeginPeriod(1) == 0
    try:
        report = asyncio.run(run_load(config_from_args(args)))
    finally:
        if timer_period_enabled:
            ctypes.windll.winmm.timeEndPeriod(1)
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
