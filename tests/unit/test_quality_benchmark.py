from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

from scripts.quality_benchmark import (
    REQUIRED_ENTITY_TYPES,
    BenchmarkMetrics,
    Case,
    GoldEntity,
    Prediction,
    TypeScore,
    _calculate_f1,
    gate_failures,
    levenshtein,
    load_dataset,
    macro_averages,
    main,
    micro_score,
    normalized_mask_similarity,
    score_keys,
    summarize,
    validate_dataset,
    verify_dataset_hash,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks" / "pii_quality.jsonl"


def _case(
    text: str = "Дата рождения: 03.11.1985",
    entities: list[GoldEntity] | None = None,
    *,
    case_id: str = "case-001",
    tags: list[str] | None = None,
    gold_mask: str | None = None,
) -> Case:
    entities = entities or []
    if gold_mask is None:
        characters = list(text)
        for entity in entities:
            characters[entity.start : entity.end] = ["*"] * (entity.end - entity.start)
        gold_mask = "".join(characters)
    return Case(
        id=case_id,
        text=text,
        gold_entities=entities,
        gold_mask=gold_mask,
        tags=tags or ["positive"],
    )


def _entity(text: str, needle: str, entity_type: str) -> GoldEntity:
    start = text.index(needle)
    return GoldEntity(entity_type, start, start + len(needle), needle)


def _metrics(**overrides: object) -> BenchmarkMetrics:
    metrics = BenchmarkMetrics(
        case_count=1,
        positive_cases=1,
        negative_cases=0,
        gold_entity_count=1,
        micro=TypeScore(tp=1),
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        per_type={"EMAIL": TypeScore(tp=1)},
        invalid_spans=0,
        invariant_violations=0,
        negative_fp_rate=0.0,
        negative_fp_cases=0,
        exact_mask_match_rate=1.0,
        mean_mask_similarity=1.0,
        demask_exact_rate=1.0,
    )
    for key, value in overrides.items():
        setattr(metrics, key, value)
    return metrics


def _gate(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "min_mask_similarity": None,
        "min_micro_f1": None,
        "min_macro_f1": None,
        "min_demask_exact": None,
        "max_negative_fp": None,
        "max_invalid_spans": 0,
        "max_invariant_violations": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_confusion_counts_exact_span_and_type() -> None:
    gold = {("EMAIL", 0, 5), ("PHONE", 10, 15)}
    predicted = {("EMAIL", 0, 5), ("PERSON", 20, 25), ("EMAIL", 0, 6)}
    per_type = score_keys(gold, predicted)
    micro = micro_score(per_type)
    assert micro.tp == 1
    assert micro.fp == 2
    assert micro.fn == 1
    assert ("EMAIL", 0, 6) not in gold
    assert ("PHONE", 0, 5) not in predicted
    precision, recall, f1 = macro_averages(per_type)
    assert per_type["EMAIL"].tp == 1
    assert per_type["EMAIL"].fp == 1
    assert per_type["PHONE"].fn == 1
    assert per_type["PERSON"].fp == 1
    assert 0.0 < precision < 1.0
    assert 0.0 < recall < 1.0
    assert 0.0 < f1 < 1.0


def test_micro_and_macro_match_manual_example() -> None:
    per_type = score_keys({("EMAIL", 0, 5), ("PHONE", 10, 15)}, {("EMAIL", 0, 5), ("PERSON", 20, 25)})
    micro = micro_score(per_type)
    assert (micro.tp, micro.fp, micro.fn) == (1, 1, 1)
    assert micro.precision == pytest.approx(0.5)
    assert micro.recall == pytest.approx(0.5)
    assert micro.f1 == pytest.approx(0.5)
    macro_precision, macro_recall, macro_f1 = macro_averages(per_type)
    assert macro_precision == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert macro_recall == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert macro_f1 == pytest.approx((1.0 + 0.0 + 0.0) / 3)


def test_zero_division_is_defined() -> None:
    empty = micro_score({})
    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0
    assert macro_averages({}) == (1.0, 1.0, 1.0)
    only_fp = TypeScore(fp=1)
    assert only_fp.precision == 0.0
    assert only_fp.recall == 0.0
    assert only_fp.f1 == 0.0


def test_f1_perfect() -> None:
    assert _calculate_f1(tp=1, fp=0, fn=0) == pytest.approx(1.0)
    score = TypeScore(tp=1, fp=0, fn=0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(1.0)


def test_f1_only_fp() -> None:
    assert _calculate_f1(tp=0, fp=1, fn=0) == pytest.approx(0.0)
    score = TypeScore(tp=0, fp=1, fn=0)
    assert score.precision == pytest.approx(0.0)
    assert score.recall == pytest.approx(0.0)
    assert score.f1 == pytest.approx(0.0)


def test_f1_only_fn() -> None:
    assert _calculate_f1(tp=0, fp=0, fn=1) == pytest.approx(0.0)
    score = TypeScore(tp=0, fp=0, fn=1)
    assert score.precision == pytest.approx(0.0)
    assert score.recall == pytest.approx(0.0)
    assert score.f1 == pytest.approx(0.0)


def test_f1_tp_zero_fp_and_fn_present() -> None:
    assert _calculate_f1(tp=0, fp=1, fn=1) == pytest.approx(0.0)
    score = TypeScore(tp=0, fp=1, fn=1)
    assert score.precision == pytest.approx(0.0)
    assert score.recall == pytest.approx(0.0)
    assert score.f1 == pytest.approx(0.0)


def test_f1_half_precision() -> None:
    assert _calculate_f1(tp=5, fp=5, fn=0) == pytest.approx(2.0 / 3.0)
    assert _calculate_f1(tp=5, fp=0, fn=5) == pytest.approx(2.0 / 3.0)


def test_f1_empty_gold_empty_prediction() -> None:
    per_type = score_keys(set(), set())
    assert per_type == {}
    micro = micro_score(per_type)
    assert micro.tp == 0
    assert micro.fp == 0
    assert micro.fn == 0
    assert macro_averages(per_type) == (1.0, 1.0, 1.0)


def test_type_mismatch_counts_fp_and_fn() -> None:
    gold = {("PERSON", 0, 5)}
    predicted = {("ADDRESS", 0, 5)}
    per_type = score_keys(gold, predicted)
    assert per_type["PERSON"].fn == 1
    assert per_type["PERSON"].tp == 0
    assert per_type["ADDRESS"].fp == 1
    assert per_type["ADDRESS"].tp == 0
    micro = micro_score(per_type)
    assert (micro.tp, micro.fp, micro.fn) == (0, 1, 1)
    assert micro.f1 == pytest.approx(0.0)


def test_verify_dataset_hash_matches(tmp_path: Path) -> None:
    dataset = tmp_path / "good.jsonl"
    dataset.write_bytes(b"line1\nline2\n")
    expected = hashlib.sha256(b"line1\nline2\n").hexdigest()
    verify_dataset_hash(dataset, expected)


def test_verify_dataset_hash_mismatch_raises(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_bytes(b"tampered content")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        verify_dataset_hash(dataset, "0" * 64)


def test_verify_dataset_hash_uses_module_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "default.jsonl"
    dataset.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    monkeypatch.setattr("scripts.quality_benchmark.DATASET_SHA256", digest)
    verify_dataset_hash(dataset)


def test_verify_dataset_hash_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        verify_dataset_hash(missing)


def test_levenshtein_similarity_bounds() -> None:
    assert levenshtein("abc", "abc") == 0
    assert normalized_mask_similarity("abc", "abc") == 1.0
    assert normalized_mask_similarity("", "") == 1.0
    assert normalized_mask_similarity("abc", "xyz") == 0.0
    assert levenshtein("abc", "xyz") == 3


def test_dataset_schema_offsets_and_mask() -> None:
    cases = load_dataset(DATASET)
    assert validate_dataset(cases) == []
    assert len(cases) >= 200
    tags = [tag for case in cases for tag in case.tags]
    assert sum("negative" in case.tags for case in cases) >= 40
    assert sum("mixed" in case.tags for case in cases) >= 30
    counts: dict[str, int] = {}
    for case in cases:
        for entity in case.gold_entities:
            counts[entity.type] = counts.get(entity.type, 0) + 1
            assert case.text[entity.start : entity.end] == entity.text
    assert set(counts) >= REQUIRED_ENTITY_TYPES
    assert all(count >= 8 for count in counts.values())
    assert "positive" in tags


def test_dataset_is_not_mostly_unit_test_copies() -> None:
    cases = load_dataset(DATASET)
    blobs = []
    for path in (ROOT / "tests" / "unit").glob("test_*.py"):
        if path.name in {"test_quality_benchmark.py", "test_load_check.py"}:
            continue
        blobs.append(path.read_text(encoding="utf-8"))
    blob = "\n".join(blobs)
    copies = [case.id for case in cases if case.text in blob]
    assert len(copies) / len(cases) <= 0.7


def test_duplicate_and_overlapping_gold_are_rejected() -> None:
    text = "abcdef"
    duplicate = _case(
        text,
        [
            GoldEntity("EMAIL", 0, 3, "abc"),
            GoldEntity("EMAIL", 0, 3, "abc"),
        ],
        gold_mask="***def",
    )
    overlap = _case(
        text,
        [
            GoldEntity("EMAIL", 0, 4, "abcd"),
            GoldEntity("PHONE", 2, 5, "cde"),
        ],
        gold_mask="******",
    )
    broken = _case(text, [GoldEntity("EMAIL", 0, 3, "zzz")], gold_mask="***def")
    bad_mask = _case(text, [GoldEntity("EMAIL", 0, 3, "abc")], gold_mask="abcdef")
    assert any("duplicate" in error for error in validate_dataset([duplicate]))
    assert any("overlapping" in error for error in validate_dataset([overlap]))
    assert any("gold text" in error for error in validate_dataset([broken]))
    assert any("gold_mask" in error for error in validate_dataset([bad_mask]))


def test_summarize_negative_false_positive_rate() -> None:
    text = "нет пдн"
    negative = _case(text, [], case_id="n", tags=["negative"])
    positive_text = "mail user@example.com"
    positive = _case(
        positive_text,
        [_entity(positive_text, "user@example.com", "EMAIL")],
        case_id="p",
        tags=["positive", "email"],
    )
    predictions = [
        Prediction(entities=[("EMAIL", 0, 3)], mask=text, demask_exact=True),
        Prediction(entities=[("EMAIL", 5, 21)], mask="mail " + ("*" * 16), demask_exact=True),
    ]
    metrics = summarize([negative, positive], predictions, http_mode=False)
    assert metrics.negative_fp_rate == pytest.approx(1.0)
    assert metrics.exact_mask_match_rate == pytest.approx(1.0)
    assert metrics.micro.tp == 1
    assert metrics.invalid_spans == 0


def test_cli_gate_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "tiny.jsonl"
    text = "код 123456"
    payload = {
        "id": "case-001",
        "text": text,
        "gold_entities": [],
        "gold_mask": text,
        "tags": ["negative"],
    }
    dataset.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    import hashlib

    monkeypatch.setattr(
        "scripts.quality_benchmark.DATASET_SHA256",
        hashlib.sha256(dataset.read_bytes()).hexdigest(),
    )
    args = [
        "--dataset",
        str(dataset),
        "--min-mask-similarity",
        "0.95",
        "--min-micro-f1",
        "0.95",
        "--min-macro-f1",
        "0.90",
    ]

    async def perfect(_cases: list[Case], _args: Namespace) -> BenchmarkMetrics:
        return _metrics()

    monkeypatch.setattr("scripts.quality_benchmark.run_benchmark", perfect)
    assert main(args) == 0

    async def poor(_cases: list[Case], _args: Namespace) -> BenchmarkMetrics:
        return _metrics(
            micro=TypeScore(fp=1, fn=1),
            macro_f1=0.2,
            mean_mask_similarity=0.2,
        )

    monkeypatch.setattr("scripts.quality_benchmark.run_benchmark", poor)
    assert main(args) == 1


def test_cli_rejects_invalid_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    payload = {
        "id": "case-001",
        "text": "abcdef",
        "gold_entities": [
            {"type": "EMAIL", "start": 0, "end": 3, "text": "abc"},
            {"type": "EMAIL", "start": 0, "end": 3, "text": "abc"},
        ],
        "gold_mask": "***def",
        "tags": ["positive"],
    }
    dataset.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert main(["--dataset", str(dataset)]) == 2


def test_gate_thresholds_are_explicit() -> None:
    metrics = _metrics(mean_mask_similarity=0.5, negative_fp_rate=0.2, invalid_spans=2)
    failures = gate_failures(
        metrics,
        _gate(min_mask_similarity=0.95, max_negative_fp=0.05, max_invalid_spans=0),
    )
    assert any("mask similarity" in item for item in failures)
    assert any("negative false-positive" in item for item in failures)
    assert any("invalid spans" in item for item in failures)


@pytest.mark.asyncio
async def test_http_mode_compares_mask_and_exact_source() -> None:
    from scripts.quality_benchmark import predict_http

    store: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        payload_id = body["payload_id"]
        payload = body["payload"]
        if payload_id not in store:
            store[payload_id] = payload
            return httpx.Response(200, json={"result": "*" * len(payload)})
        if payload == "*" * len(store[payload_id]):
            return httpx.Response(200, json={"result": store[payload_id]})
        return httpx.Response(200, json={"result": "broken"})

    text = "mail user@example.com"
    case = _case(text, [_entity(text, "user@example.com", "EMAIL")])
    predictions = await predict_http(
        [case],
        base_url="http://example.test",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )
    assert predictions[0].demask_exact is True
    assert predictions[0].mask == "*" * len(text)
    assert normalized_mask_similarity(predictions[0].mask, case.gold_mask) < 1.0
