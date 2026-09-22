#!/usr/bin/env python3
"""Open-loop нагрузка на POST /process: фаза маски, затем фаза демаски.

Один httpx-клиент на прогон. Слоты планируются по целевому RPS и не ждут
завершения предыдущего запроса, пока не упёрлись в semaphore (max in-flight).
Успех strict-прогона зависит от фактического RPS и latency, а не только от
отсутствия пяти 5xx подряд. 1000 RPS не считается достигнутым без отчёта прогона.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import heapq
import json
import math
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


class Clock(Protocol):
    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)
        else:
            await asyncio.sleep(0)


class VirtualClock:
    """Виртуальные часы для тестов: время прыгает к ближайшему sleep, без wall-clock."""

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
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
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
    rps: float = 1000.0
    duration: float = 30.0
    concurrency: int = 200
    timeout: float = 10.0
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
class LoadReport:
    target_rps: float
    duration: float
    strict: bool
    scheduled: int = 0
    started: int = 0
    completed: int = 0
    scheduling_rps: float = 0.0
    completion_rps: float = 0.0
    success: int = 0
    retried_requests: int = 0
    final_429: int = 0
    final_4xx: int = 0
    final_5xx: int = 0
    network_errors: int = 0
    contract_errors: int = 0
    max_consecutive_invalid: int = 0
    consecutive_invalid: int = 0
    mask_p50: float = 0.0
    mask_p95: float = 0.0
    mask_p99: float = 0.0
    mask_max: float = 0.0
    demask_p50: float = 0.0
    demask_p95: float = 0.0
    demask_p99: float = 0.0
    demask_max: float = 0.0
    fraction_latency_gt_1s: float = 0.0
    demask_mismatches: int = 0
    elapsed_seconds: float = 0.0
    peak_in_flight: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.reasons


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
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def failure_reasons(report: LoadReport, *, strict: bool) -> list[str]:
    reasons: list[str] = []
    if report.demask_mismatches:
        reasons.append(f"demask mismatches: {report.demask_mismatches}")
    if report.final_5xx:
        reasons.append(f"final 5xx: {report.final_5xx}")
    if report.final_429:
        reasons.append(f"final 429: {report.final_429}")
    if report.final_4xx:
        reasons.append(f"final 4xx: {report.final_4xx}")
    if report.network_errors:
        reasons.append(f"network errors: {report.network_errors}")
    if report.contract_errors:
        reasons.append(f"invalid JSON contract: {report.contract_errors}")
    if report.max_consecutive_invalid >= INVALID_STREAK_LIMIT:
        reasons.append(f"max consecutive invalid >= {INVALID_STREAK_LIMIT}")
    if strict:
        expected = report.target_rps * report.duration
        if report.scheduled < 0.95 * expected:
            reasons.append(f"scheduled mask requests {report.scheduled} < 95% of rps*duration ({expected:.0f})")
        if report.completion_rps < 0.95 * report.target_rps:
            reasons.append(
                f"achieved completion RPS {report.completion_rps:.2f} < 95% of target {report.target_rps:.2f}"
            )
        if report.mask_p95 > 1.0:
            reasons.append(f"mask p95 {report.mask_p95:.4f}s > 1s")
        if report.demask_p95 > 1.0:
            reasons.append(f"demask p95 {report.demask_p95:.4f}s > 1s")
    return reasons


@dataclass(slots=True)
class _RunState:
    consecutive_invalid: int = 0
    max_consecutive_invalid: int = 0
    stop: bool = False
    in_flight: int = 0
    peak_in_flight: int = 0


def _note_final(state: _RunState, kind: str) -> None:
    if kind == "success":
        state.consecutive_invalid = 0
        return
    if kind == "rate_limited":
        return
    state.consecutive_invalid += 1
    state.max_consecutive_invalid = max(state.max_consecutive_invalid, state.consecutive_invalid)
    if state.consecutive_invalid >= INVALID_STREAK_LIMIT:
        state.stop = True


def _retry_after_seconds(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _classify(response: httpx.Response) -> Outcome:
    status = response.status_code
    if status == 429:
        return Outcome(kind="rate_limited", status=429)
    if status == 200:
        try:
            body = response.json()
        except json.JSONDecodeError:
            return Outcome(kind="invalid", status=200)
        if not isinstance(body, dict) or not isinstance(body.get("result"), str):
            return Outcome(kind="invalid", status=200)
        return Outcome(kind="success", status=200, body=body["result"])
    if 400 <= status < 500:
        return Outcome(kind="invalid", status=status)
    if status >= 500:
        return Outcome(kind="invalid", status=status)
    return Outcome(kind="invalid", status=status)


async def _request_with_retries(
    client: httpx.AsyncClient,
    clock: Clock,
    state: _RunState,
    payload: str,
    payload_id: str,
    timeout: float,
) -> Outcome:
    last = Outcome(kind="invalid", status=0)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = await client.post(
                "/process",
                json={"payload": payload, "payload_id": payload_id},
                timeout=timeout,
            )
        except httpx.HTTPError:
            last = Outcome(kind="invalid", status=0, attempts=attempt)
        else:
            last = _classify(response)
            last.attempts = attempt
            if last.kind == "success" or attempt > MAX_RETRIES:
                return last
            if last.kind == "rate_limited":
                await clock.sleep(_retry_after_seconds(response))
                continue
        if last.kind == "success" or attempt > MAX_RETRIES:
            return last
        await clock.sleep(0)
    return last


async def _occupy(
    client: httpx.AsyncClient,
    clock: Clock,
    state: _RunState,
    semaphore: asyncio.Semaphore,
    payload: str,
    payload_id: str,
    timeout: float,
) -> tuple[Outcome, float]:
    """Permit уже взят планировщиком. Отпускаем его в finally, чтобы очередь была ограничена."""
    state.in_flight += 1
    state.peak_in_flight = max(state.peak_in_flight, state.in_flight)
    started = clock.monotonic()
    try:
        outcome = await _request_with_retries(client, clock, state, payload, payload_id, timeout)
        if outcome.attempts:
            _note_final(state, outcome.kind)
        return outcome, max(clock.monotonic() - started, 0.0)
    finally:
        state.in_flight -= 1
        semaphore.release()


async def run_load(
    config: LoadConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> LoadReport:
    clock = clock or SystemClock()
    virtual = clock if isinstance(clock, VirtualClock) else None
    if virtual is not None:
        virtual.start()
    report = LoadReport(target_rps=config.rps, duration=config.duration, strict=config.strict)
    try:
        await _execute(config, report, transport=transport, clock=clock)
    finally:
        if virtual is not None:
            await virtual.stop()
    report.reasons = failure_reasons(report, strict=config.strict)
    return report


async def _execute(
    config: LoadConfig,
    report: LoadReport,
    *,
    transport: httpx.AsyncBaseTransport | None,
    clock: Clock,
) -> None:
    max_connections = config.max_connections or max(config.concurrency, 1)
    keepalive = config.max_keepalive_connections or max_connections
    keepalive = min(keepalive, max_connections)
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=keepalive)
    semaphore = asyncio.Semaphore(max(config.concurrency, 1))
    state = _RunState()
    mask_latencies: list[float] = []
    demask_latencies: list[float] = []
    saved: list[tuple[str, str, str]] = []

    async with httpx.AsyncClient(
        base_url=config.base_url,
        limits=limits,
        timeout=config.timeout,
        transport=transport,
    ) as client:
        phase_started = clock.monotonic()
        slot_count = planned_slot_count(config.duration, config.rps)
        tasks: list[asyncio.Task[tuple[Outcome, float, str, str]]] = []

        async def launch(index: int) -> tuple[Outcome, float, str, str]:
            report.started += 1
            payload = SYNTHETIC_PAYLOADS[index % len(SYNTHETIC_PAYLOADS)]
            payload_id = uuid.uuid4().hex
            outcome, latency = await _occupy(
                client,
                clock,
                state,
                semaphore,
                payload,
                payload_id,
                config.timeout,
            )
            return outcome, latency, payload_id, payload

        for index in range(slot_count):
            if state.stop:
                break
            target = phase_started + index / config.rps
            delay = target - clock.monotonic()
            if delay > 0:
                await clock.sleep(delay)
            if state.stop:
                break
            await semaphore.acquire()
            if state.stop:
                semaphore.release()
                break
            report.scheduled += 1
            tasks.append(asyncio.create_task(launch(index)))

        schedule_elapsed = max(clock.monotonic() - phase_started, 0.0)
        results = await asyncio.gather(*tasks) if tasks else []
        mask_completed = len(results)
        mask_elapsed = max(clock.monotonic() - phase_started, 1e-9)
        report.scheduling_rps = report.scheduled / max(schedule_elapsed, 1e-9)
        report.completed = mask_completed
        report.completion_rps = mask_completed / mask_elapsed

        for outcome, latency, payload_id, payload in results:
            if outcome.attempts > 1:
                report.retried_requests += 1
            if outcome.kind == "success" and outcome.body is not None:
                report.success += 1
                mask_latencies.append(latency)
                saved.append((payload_id, outcome.body, payload))
            else:
                _count_failure(report, outcome)
                if outcome.attempts:
                    mask_latencies.append(latency)

        if not state.stop:
            demask_latencies, mismatches = await _demask_phase(
                client,
                clock,
                state,
                semaphore,
                config,
                report,
                saved,
            )
            report.demask_mismatches = mismatches

        report.elapsed_seconds = max(clock.monotonic() - phase_started, 0.0)
        report.max_consecutive_invalid = state.max_consecutive_invalid
        report.consecutive_invalid = state.consecutive_invalid
        report.peak_in_flight = state.peak_in_flight
        report.mask_p50 = percentile(mask_latencies, 50)
        report.mask_p95 = percentile(mask_latencies, 95)
        report.mask_p99 = percentile(mask_latencies, 99)
        report.mask_max = max(mask_latencies, default=0.0)
        report.demask_p50 = percentile(demask_latencies, 50)
        report.demask_p95 = percentile(demask_latencies, 95)
        report.demask_p99 = percentile(demask_latencies, 99)
        report.demask_max = max(demask_latencies, default=0.0)
        if mask_latencies:
            report.fraction_latency_gt_1s = sum(item > 1.0 for item in mask_latencies) / len(mask_latencies)


def _count_failure(report: LoadReport, outcome: Outcome) -> None:
    if outcome.kind == "rate_limited":
        report.final_429 += 1
        return
    if outcome.status == 0 and outcome.attempts:
        report.network_errors += 1
        return
    if outcome.status == 200:
        report.contract_errors += 1
        return
    if 400 <= outcome.status < 500:
        report.final_4xx += 1
        return
    if outcome.status >= 500 or outcome.status == 0:
        report.final_5xx += 1


async def _demask_phase(
    client: httpx.AsyncClient,
    clock: Clock,
    state: _RunState,
    semaphore: asyncio.Semaphore,
    config: LoadConfig,
    report: LoadReport,
    saved: list[tuple[str, str, str]],
) -> tuple[list[float], int]:
    latencies: list[float] = []
    mismatches = 0

    tasks: list[asyncio.Task[None]] = []

    async def one(payload_id: str, masked: str, original: str) -> None:
        nonlocal mismatches
        outcome, latency = await _occupy(
            client,
            clock,
            state,
            semaphore,
            masked,
            payload_id,
            config.timeout,
        )
        latencies.append(latency)
        if outcome.attempts > 1:
            report.retried_requests += 1
        if outcome.kind != "success" or outcome.body != original:
            if outcome.kind != "success":
                _count_failure(report, outcome)
            mismatches += 1

    for payload_id, masked, original in saved:
        if state.stop:
            break
        await semaphore.acquire()
        if state.stop:
            semaphore.release()
            break
        tasks.append(asyncio.create_task(one(payload_id, masked, original)))
    if tasks:
        await asyncio.gather(*tasks)
    return latencies, mismatches


def print_report(report: LoadReport) -> None:
    print(f"target_rps: {report.target_rps}")
    print(f"duration: {report.duration}")
    print(f"scheduled_requests: {report.scheduled}")
    print(f"started_requests: {report.started}")
    print(f"completed_requests: {report.completed}")
    print(f"achieved_scheduling_rps: {report.scheduling_rps:.2f}")
    print(f"achieved_completion_rps: {report.completion_rps:.2f}")
    print(f"success: {report.success}")
    print(f"retried_requests: {report.retried_requests}")
    print(f"final_429: {report.final_429}")
    print(f"final_4xx: {report.final_4xx}")
    print(f"final_5xx: {report.final_5xx}")
    print(f"network_errors: {report.network_errors}")
    print(f"contract_errors: {report.contract_errors}")
    print(f"max_consecutive_invalid: {report.max_consecutive_invalid}")
    print(f"mask_latency_p50: {report.mask_p50:.4f}")
    print(f"mask_latency_p95: {report.mask_p95:.4f}")
    print(f"mask_latency_p99: {report.mask_p99:.4f}")
    print(f"mask_latency_max: {report.mask_max:.4f}")
    print(f"demask_latency_p50: {report.demask_p50:.4f}")
    print(f"demask_latency_p95: {report.demask_p95:.4f}")
    print(f"demask_latency_p99: {report.demask_p99:.4f}")
    print(f"demask_latency_max: {report.demask_max:.4f}")
    print(f"fraction_latency_gt_1s: {report.fraction_latency_gt_1s:.4f}")
    print(f"demask_mismatches: {report.demask_mismatches}")
    print(f"elapsed_seconds: {report.elapsed_seconds:.4f}")
    print(f"peak_in_flight: {report.peak_in_flight}")
    for reason in report.reasons:
        print(f"fail_reason: {reason}")
    print(f"result: {'PASS' if report.passed else 'FAIL'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop load check for PII masking gateway")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rps", type=float, default=1000.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--max-keepalive-connections", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail unless measured RPS and latency meet the gate")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LoadConfig:
    return LoadConfig(
        base_url=args.base_url,
        rps=args.rps,
        duration=args.duration,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
        strict=args.strict,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run_load(config_from_args(args)))
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
