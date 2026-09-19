"""Phase 6A cross-encoder reranking over existing dense candidates."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import CrossEncoder, SentenceTransformer

from materials_rag.ingestion.dense_retrieval import (
    MODEL_NAME as BI_ENCODER_MODEL_NAME,
)
from materials_rag.ingestion.dense_retrieval import (
    QUERY_PREFIX,
    _encode_texts,
    compute_metrics,
)
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
from materials_rag.ingestion.public_sanity import PUBLIC_SANITY_QUESTIONS
from materials_rag.ingestion.qdrant_retrieval import COLLECTION_NAME, open_qdrant, qdrant_search
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_HF_REPOSITORY = "https://huggingface.co/BAAI/bge-reranker-base"
DENSE_CANDIDATE_K = 20
RERANK_BATCH_SIZE = 8
KNOWN_CASE_EVIDENCE_IDS = {
    "SYNQ-001-A": ["FAT-B017#results", "FAT-B017#conditions"],
    "SYNQ-012-A": ["RCA-B017-004-rev2#status", "RCA-B017-004-rev2#conclusion"],
}


@dataclass
class Phase6AResult:
    reranked_results: list[dict[str, Any]]
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


def _model_revision(model: CrossEncoder) -> str | None:
    try:
        value = getattr(model.model.config, "_commit_hash", None)
        if value:
            return str(value)
    except AttributeError:
        return None
    try:
        return str(model.model.config._name_or_path)
    except AttributeError:
        return None


def _model_cache_location(model_name: str, revision: str | None) -> str | None:
    if not revision:
        return None
    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / model_name.replace("/", "--")
    snapshot = snapshot.with_name(f"models--{snapshot.name}") / "snapshots" / revision
    return snapshot.as_posix() if snapshot.exists() else None


def _model_dtype(model: CrossEncoder) -> str:
    try:
        return str(next(model.model.parameters()).dtype)
    except StopIteration:
        return "unknown"


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """Open the already cached reranker; this must not substitute another model."""

    model = CrossEncoder(model_name, local_files_only=True)
    model.model.eval()
    return model


def _reranker_metadata(model: CrossEncoder, repo_root: Path) -> dict[str, Any]:
    revision = _model_revision(model)
    cache_location = _model_cache_location(RERANKER_MODEL_NAME, revision)
    return {
        "model_name": RERANKER_MODEL_NAME,
        "huggingface_repository": RERANKER_HF_REPOSITORY,
        "resolved_revision": revision,
        "local_cache_location": cache_location,
        "local_cache_location_relative_to_repo": (
            _relative(Path(cache_location), repo_root) if cache_location else None
        ),
        "local_files_only_verified": True,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(model.device),
        "dtype": _model_dtype(model),
        "max_sequence_length": model.max_seq_length,
        "batch_size": RERANK_BATCH_SIZE,
    }


def _score_pairs(model: CrossEncoder, pairs: list[tuple[str, str]]) -> list[float]:
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
        raise ValueError("Reranker produced non-finite scores.")
    return [float(score) for score in scores]


def _rerank_one_question(
    question: dict[str, Any],
    dense_row: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    rerank_scores: list[float],
) -> dict[str, Any]:
    candidates = dense_row["results"][:DENSE_CANDIDATE_K]
    if len(candidates) != DENSE_CANDIDATE_K:
        raise ValueError(
            f"{question['question_id']} has {len(candidates)} dense candidates, expected "
            f"{DENSE_CANDIDATE_K}."
        )
    dense_items: list[dict[str, Any]] = []
    for candidate, rerank_score in zip(candidates, rerank_scores, strict=True):
        unit = units_by_id[candidate["retrieval_unit_id"]]
        metadata = unit.get("metadata", {})
        dense_items.append(
            {
                "question_id": question["question_id"],
                "retrieval_unit_id": candidate["retrieval_unit_id"],
                "dense_rank": int(candidate["rank"]),
                "dense_score": float(candidate["score"]),
                "rerank_score": float(rerank_score),
                "kind": unit["kind"],
                "source": unit["source"],
                "evidence_id": metadata.get("evidence_id"),
                "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
            }
        )
    reranked = sorted(dense_items, key=lambda row: (-row["rerank_score"], row["dense_rank"]))
    for rank, row in enumerate(reranked, start=1):
        row["rerank_rank"] = rank
    return {
        "question_id": question["question_id"],
        "query": question["query"],
        "candidate_k": DENSE_CANDIDATE_K,
        "results": reranked,
    }


def rerank_dense_candidates(
    questions: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    model: CrossEncoder,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    dense_by_question = {row["question_id"]: row for row in dense_records}
    reranked_records: list[dict[str, Any]] = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        if progress_callback is not None:
            progress_callback(index, total, question["question_id"])
        dense_row = dense_by_question[question["question_id"]]
        pairs = [
            (question["query"], str(units_by_id[candidate["retrieval_unit_id"]]["retrieval_text"]))
            for candidate in dense_row["results"][:DENSE_CANDIDATE_K]
        ]
        scores = _score_pairs(model, pairs)
        reranked_records.append(_rerank_one_question(question, dense_row, units_by_id, scores))
    return reranked_records


def _results_by_question_for_metrics(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["question_id"]: [
            {
                "retrieval_unit_id": result["retrieval_unit_id"],
                "score": result["rerank_score"],
            }
            for result in sorted(row["results"], key=lambda item: item["rerank_rank"])
        ]
        for row in records
    }


def _comparison_table(
    dense_metrics: dict[str, Any],
    reranked_metrics: dict[str, Any],
) -> list[dict[str, float]]:
    dense = dense_metrics["overall_non_abstain"]
    reranked = reranked_metrics["overall_non_abstain"]
    ordered = ["hit@1", "hit@3", "hit@5", "hit@10", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]
    return [
        {
            "metric": metric,
            "dense_baseline": float(dense[metric]),
            "reranked": float(reranked[metric]),
            "delta": float(reranked[metric] - dense[metric]),
        }
        for metric in ordered
    ]


def _candidate_recall_ceiling(
    questions: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_by_question = {row["question_id"]: row for row in dense_records}
    per_question: list[dict[str, Any]] = []
    for question in questions:
        gold_ids = list(question.get("gold_chunk_ids", []))
        dense_ids = [row["retrieval_unit_id"] for row in dense_by_question[question["question_id"]]["results"][:DENSE_CANDIDATE_K]]
        hit_ids = [unit_id for unit_id in gold_ids if unit_id in set(dense_ids)]
        missing_ids = [unit_id for unit_id in gold_ids if unit_id not in set(dense_ids)]
        recall = (len(hit_ids) / len(gold_ids)) if gold_ids else None
        per_question.append(
            {
                "question_id": question["question_id"],
                "answerability": question.get("answerability"),
                "gold_chunk_count": len(gold_ids),
                "gold_in_dense_top20": hit_ids,
                "gold_missing_from_dense_top20": missing_ids,
                "candidate_recall_at20": recall,
                "all_gold_in_dense_top20": bool(gold_ids) and not missing_ids,
                "some_gold_below_rank20": bool(missing_ids),
            }
        )
    with_gold = [row for row in per_question if row["gold_chunk_count"] > 0]
    return {
        "candidate_k": DENSE_CANDIDATE_K,
        "description": "Gold candidate Recall@20 before reranking; this is the maximum recoverable evidence set for Phase 6A.",
        "mean_recall_at20_all_questions_with_gold": float(
            sum(row["candidate_recall_at20"] for row in with_gold if row["candidate_recall_at20"] is not None)
            / len(with_gold)
        ),
        "questions_with_all_gold_in_dense_top20": [
            row["question_id"] for row in with_gold if row["all_gold_in_dense_top20"]
        ],
        "questions_with_some_gold_below_rank20": [
            row["question_id"] for row in with_gold if row["some_gold_below_rank20"] and row["gold_in_dense_top20"]
        ],
        "questions_with_no_gold_in_dense_top20": [
            row["question_id"] for row in with_gold if row["some_gold_below_rank20"] and not row["gold_in_dense_top20"]
        ],
        "questions_impossible_for_reranking_alone_to_fully_solve": [
            row["question_id"] for row in with_gold if row["some_gold_below_rank20"]
        ],
        "per_question": per_question,
    }


def _rank_for_evidence(row: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    for result in row["results"]:
        if result.get("evidence_id") == evidence_id:
            return {
                "retrieval_unit_id": result["retrieval_unit_id"],
                "dense_rank": result["dense_rank"],
                "rerank_rank": result["rerank_rank"],
                "dense_score": result["dense_score"],
                "rerank_score": result["rerank_score"],
            }
    return None


def _known_case_rank_changes(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    return {
        question_id: {
            evidence_id: _rank_for_evidence(by_question[question_id], evidence_id)
            for evidence_id in evidence_ids
        }
        for question_id, evidence_ids in KNOWN_CASE_EVIDENCE_IDS.items()
    }


def _packet_for_reranked_question(
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
        "retrieval_backend": "Phase 6A cross-encoder reranked dense Top-20",
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def _build_generation_packets(
    reranked_records: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> list[dict[str, Any]]:
    packets = [_packet_for_reranked_question(row, units_by_id, prompt_hash) for row in reranked_records]
    validation = validate_packets(packets, units_by_id)
    if not validation["passed"]:
        raise ValueError(f"Phase 6A generation packet leakage validation failed: {validation}")
    return packets


def _dense_top20_for_public_sanity(repo_root: Path) -> list[dict[str, Any]]:
    storage_path = repo_root / "data/processed/phase5b/qdrant_storage"
    if not storage_path.exists():
        raise FileNotFoundError(
            "Phase 5B Qdrant storage is required for public sanity reranking. "
            "Run Phase 5B before Phase 6A."
        )
    model = SentenceTransformer(BI_ENCODER_MODEL_NAME, local_files_only=True)
    model.eval()
    query_matrix = _encode_texts(
        model,
        [QUERY_PREFIX + question["query"] for question in PUBLIC_SANITY_QUESTIONS],
    )
    client = open_qdrant(storage_path)
    try:
        return [
            {
                "question_id": question["question_id"],
                "results": qdrant_search(client, COLLECTION_NAME, query_matrix[index], DENSE_CANDIDATE_K),
            }
            for index, question in enumerate(PUBLIC_SANITY_QUESTIONS)
        ]
    finally:
        client.close()


def _rerank_public_sanity(
    repo_root: Path,
    units_by_id: dict[str, dict[str, Any]],
    model: CrossEncoder,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dense_records = _dense_top20_for_public_sanity(repo_root)
    questions = [
        {
            "question_id": question["question_id"],
            "query": question["query"],
            "expected_retrieval_unit_ids": question["expected_retrieval_unit_ids"],
            "source_type": question["source_type"],
        }
        for question in PUBLIC_SANITY_QUESTIONS
    ]
    reranked = rerank_dense_candidates(questions, dense_records, units_by_id, model, progress_callback)

    report_rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    improvements: dict[str, Any] = {}
    for question, dense_row, reranked_row in zip(questions, dense_records, reranked, strict=True):
        expected = set(question["expected_retrieval_unit_ids"])
        dense_ids = [row["retrieval_unit_id"] for row in dense_row["results"]]
        rerank_ids = [row["retrieval_unit_id"] for row in sorted(reranked_row["results"], key=lambda item: item["rerank_rank"])]
        dense_expected_ranks = [index + 1 for index, unit_id in enumerate(dense_ids) if unit_id in expected]
        rerank_expected_ranks = [index + 1 for index, unit_id in enumerate(rerank_ids) if unit_id in expected]
        dense_best = min(dense_expected_ranks) if dense_expected_ranks else None
        rerank_best = min(rerank_expected_ranks) if rerank_expected_ranks else None
        row = {
            "question_id": question["question_id"],
            "query": question["query"],
            "source_type": question["source_type"],
            "expected_retrieval_unit_ids": sorted(expected),
            "dense_top5_ids": dense_ids[:GENERATION_TOP_K],
            "reranked_top5_ids": rerank_ids[:GENERATION_TOP_K],
            "best_expected_dense_rank": dense_best,
            "best_expected_rerank_rank": rerank_best,
            "expected_in_dense_top1": dense_best == 1,
            "expected_in_dense_top5": dense_best is not None and dense_best <= GENERATION_TOP_K,
            "expected_in_reranked_top1": rerank_best == 1,
            "expected_in_reranked_top5": rerank_best is not None and rerank_best <= GENERATION_TOP_K,
            "expected_source_rank_improved": (
                dense_best is not None and rerank_best is not None and rerank_best < dense_best
            ),
        }
        report_rows.append(row)
        if question["question_id"] in {"PUBSAN-003", "PUBSAN-005"}:
            improvements[question["question_id"]] = row
        if row["expected_in_dense_top5"] and not row["expected_in_reranked_top5"]:
            regressions.append(row)
    report = {
        "question_count": len(questions),
        "candidate_k": DENSE_CANDIDATE_K,
        "reranked_top_k_inspected": GENERATION_TOP_K,
        "pubsan_003_and_005": improvements,
        "regressions_from_dense_top5_success": regressions,
        "per_question": report_rows,
    }
    return reranked, report


def _excluded_source_path_violations(records: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for record in records:
        for result in record["results"]:
            source_path = str(result.get("source", {}).get("source_path", ""))
            if any(source_path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
                violations.append(f"{record['question_id']}:{result['retrieval_unit_id']}:{source_path}")
    return violations


def _metric_json(
    dense_metrics: dict[str, Any],
    reranked_metrics: dict[str, Any],
    candidate_ceiling: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "6A",
        "baseline": "Phase 5A/5B dense Top-20 candidates",
        "reranked": reranked_metrics,
        "dense_baseline": dense_metrics,
        "comparison_table": _comparison_table(dense_metrics, reranked_metrics),
        "candidate_recall_ceiling": candidate_ceiling,
    }


def run_phase6a(
    repo_root: Path | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> Phase6AResult:
    """Rerank existing dense Top-20 candidates with BAAI/bge-reranker-base."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / "data/processed/phase6a"
    output_root.mkdir(parents=True, exist_ok=True)

    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    dense_results_path = repo_root / "data/processed/phase5b/qdrant_results.jsonl"
    dense_metrics_path = repo_root / "data/processed/phase5a/metrics.json"
    prompt_path = repo_root / PROMPT_PATH
    reranked_results_path = output_root / "reranked_results.jsonl"
    metrics_path = output_root / "metrics.json"
    generation_packets_path = output_root / "generation_packets.jsonl"
    public_results_path = output_root / "public_sanity_reranked_results.jsonl"
    public_report_path = output_root / "public_sanity_report.json"

    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    dense_records = _read_jsonl(dense_results_path)
    dense_metrics = _read_json(dense_metrics_path)
    prompt_hash = _sha256_text(prompt_path.read_text(encoding="utf-8"))
    units_by_id = {unit["id"]: unit for unit in units}

    model = load_reranker()
    model_metadata = _reranker_metadata(model, repo_root)
    if (
        reranked_results_path.exists()
        and metrics_path.exists()
        and generation_packets_path.exists()
    ):
        reranked_records = _read_jsonl(reranked_results_path)
        metrics_payload = _read_json(metrics_path)
        generation_packets = _read_jsonl(generation_packets_path)
    else:
        benchmark_progress = (
            (
                lambda index, total, question_id: progress_callback(
                    "benchmark", index, total, question_id
                )
            )
            if progress_callback is not None
            else None
        )
        reranked_records = rerank_dense_candidates(
            questions, dense_records, units_by_id, model, benchmark_progress
        )
        source_violations = _excluded_source_path_violations(reranked_records)
        if source_violations:
            raise ValueError(
                f"Excluded authoring/eval source entered Phase 6A candidates: {source_violations}"
            )
        _write_jsonl(reranked_results_path, reranked_records)

        reranked_metrics = compute_metrics(questions, _results_by_question_for_metrics(reranked_records))
        candidate_ceiling = _candidate_recall_ceiling(questions, dense_records)
        metrics_payload = _metric_json(dense_metrics, reranked_metrics, candidate_ceiling)
        _write_json(metrics_path, metrics_payload)

        generation_packets = _build_generation_packets(reranked_records, units_by_id, prompt_hash)
        _write_jsonl(generation_packets_path, generation_packets)
    source_violations = _excluded_source_path_violations(reranked_records)
    if source_violations:
        raise ValueError(
            f"Excluded authoring/eval source entered Phase 6A candidates: {source_violations}"
        )
    reranked_metrics = metrics_payload["reranked"]
    candidate_ceiling = metrics_payload["candidate_recall_ceiling"]

    public_progress = (
        (
            lambda index, total, question_id: progress_callback(
                "public_sanity", index, total, question_id
            )
        )
        if progress_callback is not None
        else None
    )
    if public_results_path.exists() and public_report_path.exists():
        public_report = _read_json(public_report_path)
    else:
        public_reranked, public_report = _rerank_public_sanity(
            repo_root, units_by_id, model, public_progress
        )
        _write_jsonl(public_results_path, public_reranked)
        _write_json(public_report_path, public_report)

    known_case_rank_changes = _known_case_rank_changes(reranked_records)
    output_paths = [
        reranked_results_path,
        metrics_path,
        generation_packets_path,
        public_results_path,
        public_report_path,
    ]
    manifest_path = repo_root / "data/processed/manifests/phase_6a_manifest.json"
    manifest = {
        "phase": "6A",
        "version": "1.0.0",
        "objective": "Cross-encoder reranking of existing dense Top-20 candidates",
        "scope": {
            "candidate_source": "Phase 5B Qdrant results matching Phase 5A dense exact search",
            "candidate_k": DENSE_CANDIDATE_K,
            "generation_top_k": GENERATION_TOP_K,
            "changed_dense_embeddings": False,
            "added_candidates": False,
            "bm25_or_hybrid": False,
            "query_rewriting_or_decomposition": False,
            "llm_generation_called": False,
        },
        "reranker": model_metadata,
        "counts": {
            "benchmark_questions_reranked": len(questions),
            "benchmark_query_document_pairs_scored": len(questions) * DENSE_CANDIDATE_K,
            "public_sanity_questions_reranked": len(PUBLIC_SANITY_QUESTIONS),
            "public_sanity_query_document_pairs_scored": len(PUBLIC_SANITY_QUESTIONS) * DENSE_CANDIDATE_K,
        },
        "input_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path)
            for path in [questions_path, units_path, dense_results_path, dense_metrics_path, prompt_path]
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in output_paths
        },
        "metrics": reranked_metrics["overall_non_abstain"],
        "metric_deltas": metrics_payload["comparison_table"],
        "candidate_recall_ceiling": {
            key: value for key, value in candidate_ceiling.items() if key != "per_question"
        },
        "hard_negative_analysis": reranked_metrics["hard_negative_analysis"],
        "known_case_rank_changes": known_case_rank_changes,
        "public_sanity_report": public_report,
        "leakage_validation": validate_packets(generation_packets, units_by_id),
    }
    _write_json(manifest_path, manifest)
    return Phase6AResult(
        reranked_results=reranked_records,
        metrics=metrics_payload,
        manifest=manifest,
        outputs=output_paths,
        manifest_path=manifest_path,
    )


__all__ = [
    "DENSE_CANDIDATE_K",
    "RERANKER_MODEL_NAME",
    "load_reranker",
    "run_phase6a",
]


