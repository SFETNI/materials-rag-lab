"""Phase 7B cross-encoder reranking over Phase 7A Hybrid Top-50 candidates."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from materials_rag.ingestion.cross_encoder_reranking import (
    RERANK_BATCH_SIZE,
    RERANKER_MODEL_NAME,
    load_reranker,
)
from materials_rag.ingestion.dense_retrieval import compute_metrics
from materials_rag.ingestion.generation_packets import (
    PROMPT_PATH,
    SYSTEM_PROMPT_VERSION,
    _context_item,
    _packet_hash,
    _sha256_text,
    format_generator_input,
    validate_packets,
)
from materials_rag.ingestion.generation_packets import (
    TOP_K as GENERATION_TOP_K,
)
from materials_rag.ingestion.markdown_parser import EXCLUDED_PATH_PREFIXES
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

HYBRID_CANDIDATE_K = 50
KNOWN_CASES = {
    "SYNQ-001-A": ["FAT-B017#results", "FAT-B017#conditions"],
    "SYNQ-012-A": ["RCA-B017-004-rev2#status", "RCA-B017-004-rev2#conclusion"],
}
PUBSAN_005_EXPECTED_IDS = {
    "chunk|document_nist_in718_Kafka_2023_contour_fatigue|6",
    "process_record|nist_in718|HIP_cycles|cycle_1",
}


@dataclass
class Phase7BResult:
    hybrid_reranked_results: list[dict[str, Any]]
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    outputs: list[Path]
    manifest_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _score_pairs(model: Any, pairs: list[tuple[str, str]]) -> list[float]:
    raw_scores = model.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != len(pairs):
        raise ValueError(f"Expected {len(pairs)} reranker scores, got {scores.shape[0]}.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Hybrid+CE reranker produced non-finite scores.")
    return [float(score) for score in scores]


def _top_hybrid_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = sorted(row["results"], key=lambda result: result["hybrid_rank"])[:HYBRID_CANDIDATE_K]
    if len(candidates) != HYBRID_CANDIDATE_K:
        raise ValueError(
            f"{row['question_id']} has {len(candidates)} hybrid candidates, expected "
            f"{HYBRID_CANDIDATE_K}."
        )
    return candidates


def rerank_hybrid_candidates(
    hybrid_records: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    model: Any,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(hybrid_records)
    for index, row in enumerate(hybrid_records, start=1):
        if progress_callback is not None:
            progress_callback(index, total, row["question_id"])
        candidates = _top_hybrid_candidates(row)
        pairs = [
            (row["query"], str(units_by_id[candidate["retrieval_unit_id"]]["retrieval_text"]))
            for candidate in candidates
        ]
        scores = _score_pairs(model, pairs)
        scored_results = []
        for candidate, rerank_score in zip(candidates, scores, strict=True):
            scored_results.append({**candidate, "rerank_score": rerank_score})
        scored_results.sort(key=lambda result: (-result["rerank_score"], result["hybrid_rank"]))
        for rank, result in enumerate(scored_results, start=1):
            result["rerank_rank"] = rank
        records.append(
            {
                "question_id": row["question_id"],
                "query": row["query"],
                "hybrid_candidate_k": HYBRID_CANDIDATE_K,
                "results": scored_results,
            }
        )
    return records


def _ranked_for_metrics(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["question_id"]: [
            {"retrieval_unit_id": result["retrieval_unit_id"], "score": result["rerank_score"]}
            for result in sorted(row["results"], key=lambda result: result["rerank_rank"])
        ]
        for row in records
    }


def _comparison_table(
    dense_metrics: dict[str, Any],
    dense_ce_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
    hybrid_ce_metrics: dict[str, Any],
) -> list[dict[str, float]]:
    ordered = [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@10",
    ]
    dense = dense_metrics["overall_non_abstain"]
    dense_ce = dense_ce_metrics["overall_non_abstain"]
    hybrid = hybrid_metrics["overall_non_abstain"]
    hybrid_ce = hybrid_ce_metrics["overall_non_abstain"]
    return [
        {
            "metric": metric,
            "dense": float(dense[metric]),
            "dense_cross_encoder": float(dense_ce[metric]),
            "hybrid_rrf": float(hybrid[metric]),
            "hybrid_rrf_cross_encoder": float(hybrid_ce[metric]),
        }
        for metric in ordered
    ]


def _candidate_ceiling(
    questions: list[dict[str, Any]],
    hybrid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    hybrid_by_question = {row["question_id"]: row for row in hybrid_records}
    per_question: list[dict[str, Any]] = []
    recalls: list[float] = []
    for question in questions:
        gold_ids = list(question.get("gold_chunk_ids", []))
        candidate_ids = {
            result["retrieval_unit_id"]
            for result in _top_hybrid_candidates(hybrid_by_question[question["question_id"]])
        }
        hits = [unit_id for unit_id in gold_ids if unit_id in candidate_ids]
        missing = [unit_id for unit_id in gold_ids if unit_id not in candidate_ids]
        recall = len(hits) / len(gold_ids) if gold_ids else None
        if recall is not None:
            recalls.append(recall)
        per_question.append(
            {
                "question_id": question["question_id"],
                "gold_count": len(gold_ids),
                "gold_in_hybrid_top50": hits,
                "gold_missing_from_hybrid_top50": missing,
                "recall_at50": recall,
                "all_gold_in_hybrid_top50": bool(gold_ids) and not missing,
            }
        )
    with_gold = [row for row in per_question if row["gold_count"] > 0]
    return {
        "candidate_k": HYBRID_CANDIDATE_K,
        "mean_recall_at50_all_questions_with_gold": float(sum(recalls) / len(recalls)),
        "questions_with_all_gold_in_hybrid_top50": [
            row["question_id"] for row in with_gold if row["all_gold_in_hybrid_top50"]
        ],
        "questions_with_missing_gold_outside_hybrid_top50": [
            row["question_id"] for row in with_gold if row["gold_missing_from_hybrid_top50"]
        ],
        "questions_reranking_still_cannot_fully_solve": [
            row["question_id"] for row in with_gold if row["gold_missing_from_hybrid_top50"]
        ],
        "per_question": per_question,
    }


def _find_by_evidence(row: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    return next((result for result in row["results"] if result.get("evidence_id") == evidence_id), None)


def _find_by_id(row: dict[str, Any], unit_id: str) -> dict[str, Any] | None:
    return next((result for result in row["results"] if result["retrieval_unit_id"] == unit_id), None)


def _rank_payload(result: dict[str, Any] | None, rank_key: str) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "retrieval_unit_id": result["retrieval_unit_id"],
        "rank": result.get(rank_key),
        "dense_rank": result.get("dense_rank"),
        "dense_score": result.get("dense_score"),
        "bm25_rank": result.get("bm25_rank"),
        "bm25_score": result.get("bm25_score"),
        "hybrid_rank": result.get("hybrid_rank"),
        "rrf_score": result.get("rrf_score"),
        "rerank_rank": result.get("rerank_rank"),
        "rerank_score": result.get("rerank_score"),
    }


def _known_case_report(
    phase6a_records: list[dict[str, Any]],
    phase7a_records: list[dict[str, Any]],
    phase7b_records: list[dict[str, Any]],
) -> dict[str, Any]:
    phase6a_by_question = {row["question_id"]: row for row in phase6a_records}
    phase7a_by_question = {row["question_id"]: row for row in phase7a_records}
    phase7b_by_question = {row["question_id"]: row for row in phase7b_records}
    report: dict[str, Any] = {}
    for question_id, evidence_ids in KNOWN_CASES.items():
        report[question_id] = {}
        for evidence_id in evidence_ids:
            dense_ce = _find_by_evidence(phase6a_by_question[question_id], evidence_id)
            hybrid = _find_by_evidence(phase7a_by_question[question_id], evidence_id)
            hybrid_ce = _find_by_evidence(phase7b_by_question[question_id], evidence_id)
            report[question_id][evidence_id] = {
                "dense": _rank_payload(hybrid, "dense_rank"),
                "dense_cross_encoder": _rank_payload(dense_ce, "rerank_rank"),
                "hybrid_rrf": _rank_payload(hybrid, "hybrid_rank"),
                "hybrid_rrf_cross_encoder": _rank_payload(hybrid_ce, "rerank_rank"),
            }
    return report


def _packet_for_question(
    row: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> dict[str, Any]:
    ordered = sorted(row["results"], key=lambda result: result["rerank_rank"])[:GENERATION_TOP_K]
    retrieval_unit_ids = [result["retrieval_unit_id"] for result in ordered]
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in retrieval_unit_ids]
    generator_input = format_generator_input(row["query"], context_items)
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "top_k": GENERATION_TOP_K,
        "retrieval_unit_ids": retrieval_unit_ids,
        "retrieval_backend": "Phase 7B Hybrid RRF Top-50 + cross-encoder reranking",
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def _generation_packets(
    records: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> list[dict[str, Any]]:
    packets = [_packet_for_question(row, units_by_id, prompt_hash) for row in records]
    validation = validate_packets(packets, units_by_id)
    if not validation["passed"]:
        raise ValueError(f"Phase 7B packet validation failed: {validation}")
    for packet in packets:
        lowered = packet["generator_input"].lower()
        for term in ["bm25_score", "dense_score", "rrf_score", "rerank_score", "gold_chunk"]:
            if term in lowered:
                raise ValueError(f"Generator packet exposes forbidden term: {term}")
    return packets


def _source_path_violations(records: list[dict[str, Any]]) -> list[str]:
    violations = []
    for row in records:
        for result in row["results"]:
            source_path = str(result.get("source", {}).get("source_path", ""))
            if any(source_path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
                violations.append(f"{row['question_id']}:{result['retrieval_unit_id']}:{source_path}")
    return violations


def _public_sanity_report(
    phase7a_public_report: dict[str, Any],
    phase7a_public_hybrid: list[dict[str, Any]],
    phase7b_public: list[dict[str, Any]],
) -> dict[str, Any]:
    phase7b_by_question = {row["question_id"]: row for row in phase7b_public}
    rows = []
    regressions = []
    for previous in phase7a_public_report["per_question"]:
        question_id = previous["question_id"]
        expected_ids = set(previous["expected_retrieval_unit_ids"])
        current_row = phase7b_by_question[question_id]
        reranked_ids = [
            result["retrieval_unit_id"]
            for result in sorted(current_row["results"], key=lambda result: result["rerank_rank"])
        ]
        expected_ranks = [index + 1 for index, unit_id in enumerate(reranked_ids) if unit_id in expected_ids]
        best_rank = min(expected_ranks) if expected_ranks else None
        row = {
            **previous,
            "best_expected_hybrid_ce_rank": best_rank,
            "expected_in_hybrid_ce_top5": best_rank is not None and best_rank <= 5,
            "hybrid_ce_top5_ids": reranked_ids[:5],
        }
        rows.append(row)
        if previous.get("expected_in_hybrid_top5") and not row["expected_in_hybrid_ce_top5"]:
            regressions.append(row)
    pubsan_005 = next(row for row in rows if row["question_id"] == "PUBSAN-005")
    return {
        "question_count": len(rows),
        "pubsan_005": pubsan_005,
        "regressions_from_previous_hybrid_top5_success": regressions,
        "per_question": rows,
    }


def _pubsan_005_rank_progression(
    phase7a_public_hybrid: list[dict[str, Any]],
    phase7b_public: list[dict[str, Any]],
) -> dict[str, Any]:
    phase7a_row = next(row for row in phase7a_public_hybrid if row["question_id"] == "PUBSAN-005")
    phase7b_row = next(row for row in phase7b_public if row["question_id"] == "PUBSAN-005")

    def best(row: dict[str, Any], rank_key: str) -> dict[str, Any] | None:
        candidates = [
            result
            for result in row["results"]
            if result["retrieval_unit_id"] in PUBSAN_005_EXPECTED_IDS
            and result.get(rank_key) is not None
        ]
        if not candidates:
            return None
        return _rank_payload(min(candidates, key=lambda result: result[rank_key]), rank_key)

    return {
        "expected_retrieval_unit_ids": sorted(PUBSAN_005_EXPECTED_IDS),
        "dense": best(phase7a_row, "dense_rank"),
        "bm25": best(phase7a_row, "bm25_rank"),
        "hybrid_rrf": best(phase7a_row, "hybrid_rank"),
        "hybrid_rrf_cross_encoder": best(phase7b_row, "rerank_rank"),
    }


def run_phase7b(
    repo_root: Path | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> Phase7BResult:
    """Rerank Phase 7A Hybrid Top-50 candidates with the cached BGE cross-encoder."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / "data/processed/phase7b"
    output_root.mkdir(parents=True, exist_ok=True)

    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    phase6a_results_path = repo_root / "data/processed/phase6a/reranked_results.jsonl"
    phase6a_metrics_path = repo_root / "data/processed/phase6a/metrics.json"
    phase6a_manifest_path = repo_root / "data/processed/manifests/phase_6a_manifest.json"
    phase7a_hybrid_path = repo_root / "data/processed/phase7a/hybrid_results.jsonl"
    phase7a_metrics_path = repo_root / "data/processed/phase7a/metrics.json"
    phase7a_manifest_path = repo_root / "data/processed/manifests/phase_7a_manifest.json"
    phase7a_public_hybrid_path = repo_root / "data/processed/phase7a/public_sanity_hybrid_results.jsonl"
    phase7a_public_report_path = repo_root / "data/processed/phase7a/public_sanity_report.json"
    prompt_path = repo_root / PROMPT_PATH

    results_path = output_root / "hybrid_reranked_results.jsonl"
    metrics_path = output_root / "metrics.json"
    packets_path = output_root / "generation_packets.jsonl"
    public_results_path = output_root / "public_sanity_hybrid_reranked_results.jsonl"
    public_report_path = output_root / "public_sanity_report.json"

    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    units_by_id = {unit["id"]: unit for unit in units}
    phase6a_results = _read_jsonl(phase6a_results_path)
    phase6a_metrics = _read_json(phase6a_metrics_path)
    phase6a_manifest = _read_json(phase6a_manifest_path)
    phase7a_hybrid = _read_jsonl(phase7a_hybrid_path)
    phase7a_metrics = _read_json(phase7a_metrics_path)
    phase7a_public_hybrid = _read_jsonl(phase7a_public_hybrid_path)
    phase7a_public_report = _read_json(phase7a_public_report_path)
    prompt_hash = _sha256_text(prompt_path.read_text(encoding="utf-8"))

    if [row["question_id"] for row in phase7a_hybrid] != [question["question_id"] for question in questions]:
        raise ValueError("Phase 7A hybrid rows are not aligned with Phase 4 questions.")

    if results_path.exists() and public_results_path.exists():
        reranked_records = _read_jsonl(results_path)
        public_reranked_records = _read_jsonl(public_results_path)
    else:
        model = load_reranker()
        benchmark_progress = (
            (
                lambda index, total, question_id: progress_callback(
                    "benchmark", index, total, question_id
                )
            )
            if progress_callback is not None
            else None
        )
        public_progress = (
            (
                lambda index, total, question_id: progress_callback(
                    "public_sanity", index, total, question_id
                )
            )
            if progress_callback is not None
            else None
        )
        reranked_records = rerank_hybrid_candidates(
            phase7a_hybrid, units_by_id, model, benchmark_progress
        )
        public_reranked_records = rerank_hybrid_candidates(
            phase7a_public_hybrid, units_by_id, model, public_progress
        )
        _write_jsonl(results_path, reranked_records)
        _write_jsonl(public_results_path, public_reranked_records)

    source_violations = _source_path_violations(reranked_records) + _source_path_violations(
        public_reranked_records
    )
    if source_violations:
        raise ValueError(f"Excluded authoring/eval content entered Phase 7B: {source_violations}")

    hybrid_ce_metrics = compute_metrics(questions, _ranked_for_metrics(reranked_records))
    candidate_ceiling = _candidate_ceiling(questions, phase7a_hybrid)
    known_cases = _known_case_report(phase6a_results, phase7a_hybrid, reranked_records)
    public_report = _public_sanity_report(
        phase7a_public_report, phase7a_public_hybrid, public_reranked_records
    )
    pubsan_005_progression = _pubsan_005_rank_progression(
        phase7a_public_hybrid, public_reranked_records
    )
    generation_packets = _generation_packets(reranked_records, units_by_id, prompt_hash)
    _write_jsonl(packets_path, generation_packets)
    _write_json(public_report_path, public_report)

    metrics_payload = {
        "phase": "7B",
        "definitions": phase7a_metrics["definitions"],
        "dense": phase7a_metrics["dense_baseline"],
        "dense_cross_encoder": phase6a_metrics["reranked"],
        "hybrid_rrf": phase7a_metrics["hybrid_rrf"],
        "hybrid_rrf_cross_encoder": hybrid_ce_metrics,
        "comparison_table": _comparison_table(
            phase7a_metrics["dense_baseline"],
            phase6a_metrics["reranked"],
            phase7a_metrics["hybrid_rrf"],
            hybrid_ce_metrics,
        ),
        "candidate_ceiling": candidate_ceiling,
        "hard_negative_analysis": {
            "dense": phase7a_metrics["dense_baseline"]["hard_negative_analysis"],
            "dense_cross_encoder": phase6a_metrics["reranked"]["hard_negative_analysis"],
            "hybrid_rrf": phase7a_metrics["hybrid_rrf"]["hard_negative_analysis"],
            "hybrid_rrf_cross_encoder": hybrid_ce_metrics["hard_negative_analysis"],
        },
        "known_cases": known_cases,
        "public_sanity": public_report,
        "pubsan_005_rank_progression": pubsan_005_progression,
    }
    _write_json(metrics_path, metrics_payload)

    output_paths = [results_path, metrics_path, packets_path, public_results_path, public_report_path]
    manifest_path = repo_root / "data/processed/manifests/phase_7b_manifest.json"
    manifest = {
        "phase": "7B",
        "version": "1.0.0",
        "objective": "Cross-encoder reranking of the Hybrid Dense + BM25 candidate pool",
        "candidate_pool": {
            "source": "Phase 7A Hybrid RRF results",
            "hybrid_candidate_k": HYBRID_CANDIDATE_K,
            "candidate_set_changed_by_reranking": False,
        },
        "reranker": {
            "model_name": RERANKER_MODEL_NAME,
            "source_phase": "6A",
            "phase6a_recorded_config": phase6a_manifest.get("reranker", {}),
            "downloaded_new_model": False,
        },
        "counts": {
            "benchmark_questions": len(questions),
            "benchmark_query_document_pairs_scored": len(questions) * HYBRID_CANDIDATE_K,
            "public_sanity_questions": len(phase7a_public_hybrid),
            "public_sanity_query_document_pairs_scored": len(phase7a_public_hybrid)
            * HYBRID_CANDIDATE_K,
        },
        "input_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path)
            for path in [
                questions_path,
                units_path,
                phase6a_results_path,
                phase6a_metrics_path,
                phase6a_manifest_path,
                phase7a_hybrid_path,
                phase7a_metrics_path,
                phase7a_manifest_path,
                phase7a_public_hybrid_path,
                phase7a_public_report_path,
                prompt_path,
            ]
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in output_paths
        },
        "metrics": metrics_payload["comparison_table"],
        "candidate_ceiling": {
            key: value for key, value in candidate_ceiling.items() if key != "per_question"
        },
        "hard_negative_analysis": metrics_payload["hard_negative_analysis"],
        "known_cases": known_cases,
        "public_sanity": public_report,
        "pubsan_005_rank_progression": pubsan_005_progression,
        "generation_packet_policy": {
            "top_k": GENERATION_TOP_K,
            "uses_frozen_prompt": SYSTEM_PROMPT_VERSION,
            "exposes_dense_scores": False,
            "exposes_bm25_scores": False,
            "exposes_rrf_scores": False,
            "exposes_reranker_scores": False,
            "llm_generation_called": False,
        },
        "leakage_validation": validate_packets(generation_packets, units_by_id),
    }
    _write_json(manifest_path, manifest)

    return Phase7BResult(
        hybrid_reranked_results=reranked_records,
        metrics=metrics_payload,
        manifest=manifest,
        outputs=output_paths,
        manifest_path=manifest_path,
    )


__all__ = [
    "HYBRID_CANDIDATE_K",
    "rerank_hybrid_candidates",
    "run_phase7b",
]
