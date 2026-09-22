#!/usr/bin/env python3
"""Независимый quality benchmark детекции и маскирования ПДн.

Span-метрики считаются in-process через CascadeDetector и Masker.
Полная похожесть маски — proxy: 1 - Левенштейн(pred, gold) / max(len).
Это не заявка на официальную формулу «span-based Levenshtein» из PDF:
там не раскрыты точная метрика и способ агрегации порога 95%.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

REQUIRED_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "PERSON",
        "DATE_OF_BIRTH",
        "PLACE_OF_BIRTH",
        "PASSPORT",
        "CITIZENSHIP",
        "PASSPORT_ISSUER",
        "PASSPORT_DEPT_CODE",
        "PASSPORT_ISSUE_DATE",
        "DRIVER_LICENSE",
        "ADDRESS",
        "EMAIL",
        "PHONE",
        "INN",
        "BANK_CARD",
        "CARD_CVV",
        "CARD_PIN",
        "CARD_HOLDER",
    }
)

EntityKey = tuple[str, int, int]


@dataclass(slots=True)
class GoldEntity:
    type: str
    start: int
    end: int
    text: str


@dataclass(slots=True)
class Case:
    id: str
    text: str
    gold_entities: list[GoldEntity]
    gold_mask: str
    tags: list[str]


@dataclass(slots=True)
class Prediction:
    entities: list[EntityKey]
    mask: str
    demask_exact: bool
    invalid_spans: int = 0
    invariant_violations: int = 0


@dataclass(slots=True)
class TypeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)


@dataclass(slots=True)
class BenchmarkMetrics:
    case_count: int
    micro: TypeScore
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_type: dict[str, TypeScore]
    invalid_spans: int
    invariant_violations: int
    negative_fp_rate: float
    negative_cases: int
    negative_fp_cases: int
    exact_mask_match_rate: float
    mean_mask_similarity: float
    demask_exact_rate: float
    http_mode: bool = False
    notes: list[str] = field(default_factory=list)


def _ratio(numerator: int, denominator: int) -> float:
    """Пустой знаменатель — 1.0: отсутствие решений не считается ошибкой деления."""
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    if total == 0.0:
        return 1.0
    return 2.0 * precision * recall / total


def levenshtein(left: str, right: str) -> int:
    """Классическое расстояние Левенштейна по символам."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_mask_similarity(predicted: str, gold: str) -> float:
    """Proxy: 1 - distance / max(len). Одинаковые строки, включая пустые, дают 1."""
    if predicted == gold:
        return 1.0
    longest = max(len(predicted), len(gold))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(predicted, gold) / longest


def entity_keys(entities: list[GoldEntity] | list[EntityKey]) -> set[EntityKey]:
    keys: set[EntityKey] = set()
    for entity in entities:
        if isinstance(entity, GoldEntity):
            keys.add((entity.type, entity.start, entity.end))
        else:
            keys.add(entity)
    return keys


def score_keys(gold: set[EntityKey], predicted: set[EntityKey]) -> dict[str, TypeScore]:
    """Exact entity key = (type, start, end). Совпадение типа при другом спане — FN+FP."""
    per_type: dict[str, TypeScore] = {}
    types = {key[0] for key in gold | predicted}
    for entity_type in types:
        gold_type = {key for key in gold if key[0] == entity_type}
        pred_type = {key for key in predicted if key[0] == entity_type}
        per_type[entity_type] = TypeScore(
            tp=len(gold_type & pred_type),
            fp=len(pred_type - gold_type),
            fn=len(gold_type - pred_type),
        )
    return per_type


def micro_score(per_type: dict[str, TypeScore]) -> TypeScore:
    return TypeScore(
        tp=sum(item.tp for item in per_type.values()),
        fp=sum(item.fp for item in per_type.values()),
        fn=sum(item.fn for item in per_type.values()),
    )


def macro_averages(per_type: dict[str, TypeScore]) -> tuple[float, float, float]:
    """Независимое среднее по типам, у которых есть хотя бы один TP/FP/FN."""
    active = [item for item in per_type.values() if item.tp + item.fp + item.fn > 0]
    if not active:
        return 1.0, 1.0, 1.0
    precision = sum(item.precision for item in active) / len(active)
    recall = sum(item.recall for item in active) / len(active)
    f1 = sum(item.f1 for item in active) / len(active)
    return precision, recall, f1


def render_gold_mask(text: str, entities: list[GoldEntity]) -> str:
    characters = list(text)
    for entity in entities:
        characters[entity.start : entity.end] = ["*"] * (entity.end - entity.start)
    return "".join(characters)


def validate_dataset(cases: list[Case]) -> list[str]:
    """Схема, инвариант text == source[start:end], маска и запрет дублей/перекрытий."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        if not case.id:
            errors.append("case id is empty")
            continue
        if case.id in seen_ids:
            errors.append(f"{case.id}: duplicate case id")
        seen_ids.add(case.id)
        errors.extend(_validate_case(case))
    return errors


def _validate_case(case: Case) -> list[str]:
    errors: list[str] = []
    seen: set[EntityKey] = set()
    spans: list[tuple[int, int]] = []
    for entity in case.gold_entities:
        key = (entity.type, entity.start, entity.end)
        if key in seen:
            errors.append(f"{case.id}: duplicate gold entity {key}")
        seen.add(key)
        if entity.type not in REQUIRED_ENTITY_TYPES:
            errors.append(f"{case.id}: unknown entity type {entity.type}")
        if not (0 <= entity.start < entity.end <= len(case.text)):
            errors.append(f"{case.id}: gold span out of range {key}")
            continue
        if case.text[entity.start : entity.end] != entity.text:
            errors.append(f"{case.id}: gold text != source[{entity.start}:{entity.end}]")
        spans.append((entity.start, entity.end))
    spans.sort()
    for previous, current in itertools.pairwise(spans):
        if previous[1] > current[0]:
            errors.append(f"{case.id}: overlapping gold spans {previous} and {current}")
    if render_gold_mask(case.text, case.gold_entities) != case.gold_mask:
        errors.append(f"{case.id}: gold_mask does not match gold spans")
    return errors


def dataset_type_counts(cases: list[Case]) -> dict[str, int]:
    counts = dict.fromkeys(sorted(REQUIRED_ENTITY_TYPES), 0)
    for case in cases:
        for entity in case.gold_entities:
            counts[entity.type] = counts.get(entity.type, 0) + 1
    return counts


def load_dataset(path: Path) -> list[Case]:
    cases: list[Case] = []
    raw = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(_parse_case(payload, line_number))
    return cases


def _parse_case(payload: dict[str, Any], line_number: int) -> Case:
    required = ("id", "text", "gold_entities", "gold_mask", "tags")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"line {line_number}: missing fields {missing}")
    entities: list[GoldEntity] = []
    for entity in payload["gold_entities"]:
        entities.append(
            GoldEntity(
                type=str(entity["type"]),
                start=int(entity["start"]),
                end=int(entity["end"]),
                text=str(entity["text"]),
            )
        )
    return Case(
        id=str(payload["id"]),
        text=str(payload["text"]),
        gold_entities=entities,
        gold_mask=str(payload["gold_mask"]),
        tags=[str(tag) for tag in payload["tags"]],
    )


def summarize(cases: list[Case], predictions: list[Prediction], *, http_mode: bool) -> BenchmarkMetrics:
    if len(cases) != len(predictions):
        raise ValueError("predictions must align with cases")
    per_type: dict[str, TypeScore] = {}
    invalid_spans = 0
    invariant_violations = 0
    exact_matches = 0
    similarity_total = 0.0
    demask_hits = 0
    negative_cases = 0
    negative_fp_cases = 0

    for case, prediction in zip(cases, predictions, strict=True):
        scored = score_keys(entity_keys(case.gold_entities), set(prediction.entities))
        for entity_type, score in scored.items():
            bucket = per_type.setdefault(entity_type, TypeScore())
            bucket.tp += score.tp
            bucket.fp += score.fp
            bucket.fn += score.fn
        invalid_spans += prediction.invalid_spans
        invariant_violations += prediction.invariant_violations
        if prediction.mask == case.gold_mask:
            exact_matches += 1
        similarity_total += normalized_mask_similarity(prediction.mask, case.gold_mask)
        if prediction.demask_exact:
            demask_hits += 1
        if "negative" in case.tags:
            negative_cases += 1
            if prediction.entities:
                negative_fp_cases += 1

    case_count = len(cases)
    macro_precision, macro_recall, macro_f1 = macro_averages(per_type)
    notes = [
        "mean_mask_similarity is a proxy, not the official span-based Levenshtein from the PDF",
    ]
    if http_mode:
        notes.append("mask similarity and demask exact rate are measured over HTTP /process")
    else:
        notes.append("mask similarity is in-process; demask exact rate is Masker round-trip")
    return BenchmarkMetrics(
        case_count=case_count,
        micro=micro_score(per_type),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_type=dict(sorted(per_type.items())),
        invalid_spans=invalid_spans,
        invariant_violations=invariant_violations,
        negative_fp_rate=_ratio(negative_fp_cases, negative_cases) if negative_cases else 0.0,
        negative_cases=negative_cases,
        negative_fp_cases=negative_fp_cases,
        exact_mask_match_rate=_ratio(exact_matches, case_count) if case_count else 1.0,
        mean_mask_similarity=(similarity_total / case_count) if case_count else 1.0,
        demask_exact_rate=_ratio(demask_hits, case_count) if case_count else 1.0,
        http_mode=http_mode,
        notes=notes,
    )


async def predict_in_process(cases: list[Case]) -> list[Prediction]:
    from app.core.masker import masker
    from app.detectors.cascade import CascadeDetector

    cascade = CascadeDetector()
    await cascade.initialize()
    predictions: list[Prediction] = []
    for case in cases:
        matches = await cascade.detect(case.text)
        invalid_spans = 0
        invariant_violations = 0
        valid = []
        for match in matches:
            if not (0 <= match.start < match.end <= len(case.text)):
                invalid_spans += 1
                continue
            if case.text[match.start : match.end] != match.text:
                invariant_violations += 1
                continue
            valid.append(match)
        masked = masker.mask(case.text, valid)
        restored = masker.unmask(masked.masked_text, masked.spans)
        entities = [
            (entity_type, start, end)
            for (start, end, _original), entity_type in zip(masked.spans, masked.entity_types, strict=True)
        ]
        predictions.append(
            Prediction(
                entities=entities,
                mask=masked.masked_text,
                demask_exact=restored == case.text,
                invalid_spans=invalid_spans,
                invariant_violations=invariant_violations,
            )
        )
    return predictions


async def predict_http(
    cases: list[Case],
    *,
    base_url: str,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Prediction]:
    """Два вызова /process: исходник, затем полученная маска с тем же payload_id."""
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        transport=transport,
        limits=limits,
    ) as client:
        predictions: list[Prediction] = []
        for case in cases:
            payload_id = uuid.uuid4().hex
            mask, mask_ok = await _http_result(client, case.text, payload_id)
            if not mask_ok or mask is None:
                predictions.append(Prediction(entities=[], mask="", demask_exact=False))
                continue
            restored, restore_ok = await _http_result(client, mask, payload_id)
            predictions.append(
                Prediction(
                    entities=[],
                    mask=mask,
                    demask_exact=restore_ok and restored == case.text,
                )
            )
        return predictions


async def _http_result(client: httpx.AsyncClient, payload: str, payload_id: str) -> tuple[str | None, bool]:
    try:
        response = await client.post("/process", json={"payload": payload, "payload_id": payload_id})
    except httpx.HTTPError:
        return None, False
    if response.status_code != 200:
        return None, False
    try:
        body = response.json()
    except json.JSONDecodeError:
        return None, False
    result = body.get("result") if isinstance(body, dict) else None
    if not isinstance(result, str):
        return None, False
    return result, True


async def run_benchmark(cases: list[Case], args: argparse.Namespace) -> BenchmarkMetrics:
    base_url = getattr(args, "base_url", None)
    if base_url:
        span_predictions = await predict_in_process(cases)
        http_predictions = await predict_http(cases, base_url=base_url, timeout=args.timeout)
        merged: list[Prediction] = []
        for span, http in zip(span_predictions, http_predictions, strict=True):
            merged.append(
                Prediction(
                    entities=span.entities,
                    mask=http.mask,
                    demask_exact=http.demask_exact,
                    invalid_spans=span.invalid_spans,
                    invariant_violations=span.invariant_violations,
                )
            )
        return summarize(cases, merged, http_mode=True)
    return summarize(cases, await predict_in_process(cases), http_mode=False)


def gate_failures(metrics: BenchmarkMetrics, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.min_mask_similarity is not None and metrics.mean_mask_similarity < args.min_mask_similarity:
        failures.append(
            f"mean mask similarity proxy {metrics.mean_mask_similarity:.4f} < {args.min_mask_similarity:.4f}"
        )
    if args.min_micro_f1 is not None and metrics.micro.f1 < args.min_micro_f1:
        failures.append(f"micro F1 {metrics.micro.f1:.4f} < {args.min_micro_f1:.4f}")
    if args.min_macro_f1 is not None and metrics.macro_f1 < args.min_macro_f1:
        failures.append(f"macro F1 {metrics.macro_f1:.4f} < {args.min_macro_f1:.4f}")
    if args.min_demask_exact is not None and metrics.demask_exact_rate < args.min_demask_exact:
        failures.append(f"demask exact rate {metrics.demask_exact_rate:.4f} < {args.min_demask_exact:.4f}")
    if args.max_negative_fp is not None and metrics.negative_fp_rate > args.max_negative_fp:
        failures.append(f"negative false-positive rate {metrics.negative_fp_rate:.4f} > {args.max_negative_fp:.4f}")
    if args.max_invalid_spans is not None and metrics.invalid_spans > args.max_invalid_spans:
        failures.append(f"invalid spans {metrics.invalid_spans} > {args.max_invalid_spans}")
    if args.max_invariant_violations is not None and metrics.invariant_violations > args.max_invariant_violations:
        failures.append(f"invariant violations {metrics.invariant_violations} > {args.max_invariant_violations}")
    return failures


def print_report(metrics: BenchmarkMetrics) -> None:
    print(f"cases: {metrics.case_count}")
    print(f"http_mode: {str(metrics.http_mode).lower()}")
    print(
        "micro: "
        f"tp={metrics.micro.tp} fp={metrics.micro.fp} fn={metrics.micro.fn} "
        f"precision={metrics.micro.precision:.4f} recall={metrics.micro.recall:.4f} "
        f"f1={metrics.micro.f1:.4f}"
    )
    print(f"macro: precision={metrics.macro_precision:.4f} recall={metrics.macro_recall:.4f} f1={metrics.macro_f1:.4f}")
    print(f"exact_mask_match_rate: {metrics.exact_mask_match_rate:.4f}")
    print(f"mean_mask_similarity_proxy: {metrics.mean_mask_similarity:.4f}")
    print(f"demask_exact_rate: {metrics.demask_exact_rate:.4f}")
    print(f"invalid_spans: {metrics.invalid_spans}")
    print(f"invariant_violations: {metrics.invariant_violations}")
    print(
        "negative_false_positive_rate: "
        f"{metrics.negative_fp_rate:.4f} "
        f"({metrics.negative_fp_cases}/{metrics.negative_cases})"
    )
    print("per_type:")
    print(f"{'type':<22} {'tp':>6} {'fp':>6} {'fn':>6} {'precision':>10} {'recall':>10} {'f1':>10}")
    for entity_type, score in metrics.per_type.items():
        print(
            f"{entity_type:<22} {score.tp:6d} {score.fp:6d} {score.fn:6d} "
            f"{score.precision:10.4f} {score.recall:10.4f} {score.f1:10.4f}"
        )
    for note in metrics.notes:
        print(f"note: {note}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PII quality benchmark (span F1 and mask proxy)")
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL dataset path")
    parser.add_argument("--base-url", default=None, help="If set, mask/demask are measured via HTTP /process")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument("--min-mask-similarity", type=float, default=None)
    parser.add_argument("--min-micro-f1", type=float, default=None)
    parser.add_argument("--min-macro-f1", type=float, default=None)
    parser.add_argument("--min-demask-exact", type=float, default=None)
    parser.add_argument("--max-negative-fp", type=float, default=None)
    parser.add_argument("--max-invalid-spans", type=int, default=0)
    parser.add_argument("--max-invariant-violations", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = load_dataset(args.dataset)
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        print(f"dataset_error: {exc}", file=sys.stderr)
        return 2
    errors = validate_dataset(cases)
    if errors:
        for error in errors:
            print(f"dataset_error: {error}", file=sys.stderr)
        return 2
    counts = dataset_type_counts(cases)
    print("dataset_type_counts: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    metrics = asyncio.run(run_benchmark(cases, args))
    print_report(metrics)
    failures = gate_failures(metrics, args)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print("result: FAIL")
        return 1
    print("result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
