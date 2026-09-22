#!/usr/bin/env python3
"""Нагрузочный сценарий: 1000 RPS маскирования, затем демаскирование.

Использует только httpx и asyncio в одном процессе. Выводит метрики в stdout.
Exit code 0 при PASS, 1 при FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

import httpx

# Синтетические строки с ПДн (не реальные данные).
SYNTHETIC_PAYLOADS = [
    "Клиент Иванов Иван Иванович, паспорт 4509 123456",
    "Телефон: +7 (912) 345-67-89",
    "Дата рождения: 12.05.1990",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load check for PII masking gateway")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the service")
    parser.add_argument("--rps", type=int, default=1000, help="Target requests per second")
    parser.add_argument("--duration", type=float, default=30.0, help="Load phase duration in seconds")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds")
    return parser.parse_args()


async def _mask(client: httpx.AsyncClient, base_url: str, timeout: float) -> tuple[str, str, int]:
    """Отправляет один запрос маскирования. Возвращает (payload_id, masked, status)."""
    payload_id = uuid.uuid4().hex
    payload = SYNTHETIC_PAYLOADS[hash(payload_id) % len(SYNTHETIC_PAYLOADS)]
    try:
        resp = await client.post(
            f"{base_url}/process",
            json={"payload": payload, "payload_id": payload_id},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return payload_id, resp.json()["result"], 200
        return payload_id, "", resp.status_code
    except httpx.HTTPError:
        return payload_id, "", 0


async def _unmask(
    client: httpx.AsyncClient,
    base_url: str,
    payload_id: str,
    masked: str,
    original: str,
    timeout: float,
) -> bool:
    """Демаскирует сохранённую маску и сравнивает с исходником."""
    try:
        resp = await client.post(
            f"{base_url}/process",
            json={"payload": masked, "payload_id": payload_id},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return False
        return resp.json()["result"] == original
    except httpx.HTTPError:
        return False


async def run_load(base_url: str, rps: int, duration: float, timeout: float) -> int:
    total = 0
    success = 0
    status_429 = 0
    status_5xx = 0
    max_consecutive_5xx = 0
    consecutive_5xx = 0
    latencies: list[float] = []
    saved: dict[str, tuple[str, str]] = {}  # payload_id -> (masked, original)

    async with httpx.AsyncClient() as client:
        deadline = time.monotonic() + duration
        interval = 1.0 / rps
        next_slot = time.monotonic()

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_slot:
                await asyncio.sleep(next_slot - now)
            next_slot += interval

            start = time.monotonic()
            payload_id, masked, status = await _mask(client, base_url, timeout)
            latencies.append(time.monotonic() - start)

            total += 1
            if status == 200:
                success += 1
                consecutive_5xx = 0
                original = SYNTHETIC_PAYLOADS[hash(payload_id) % len(SYNTHETIC_PAYLOADS)]
                saved[payload_id] = (masked, original)
            elif status == 429:
                status_429 += 1
                consecutive_5xx = 0
            elif status >= 500 or status == 0:
                status_5xx += 1
                consecutive_5xx += 1
                max_consecutive_5xx = max(max_consecutive_5xx, consecutive_5xx)

        # Фаза демаскирования
        demask_mismatches = 0
        for payload_id, (masked, original) in saved.items():
            if not await _unmask(client, base_url, payload_id, masked, original, timeout):
                demask_mismatches += 1

    fraction_latency_gt_1s = (
        sum(1 for d in latencies if d > 1.0) / len(latencies) if latencies else 0.0
    )

    failed = demask_mismatches > 0 or max_consecutive_5xx >= 5
    result = "FAIL" if failed else "PASS"

    print(f"total_requests: {total}")
    print(f"success_requests: {success}")
    print(f"status_429: {status_429}")
    print(f"status_5xx: {status_5xx}")
    print(f"max_consecutive_5xx: {max_consecutive_5xx}")
    print(f"demask_mismatches: {demask_mismatches}")
    print(f"fraction_latency_gt_1s: {fraction_latency_gt_1s:.4f}")
    print(f"result: {result}")

    return 0 if not failed else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(run_load(args.base_url, args.rps, args.duration, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())