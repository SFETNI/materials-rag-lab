"""Phase 5A vanilla dense retrieval with exact cosine similarity."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 20
METRIC_CUTOFFS = (1, 3, 5, 10)


@dataclass
class Phase5AResult:
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    outputs: list[Path]
    manifest_path: Path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError("Embedding matrix contains zero or non-finite norms.")
    return (matrix / norms).astype(np.float32)


def _encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return _l2_normalize(np.asarray(embeddings, dtype=np.float32))


def _load_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


def _model_revision(model: SentenceTransformer) -> str | None:
    try:
        return str(model[0].auto_model.config._name_or_path)
    except (AttributeError, IndexError):
        return None


def _index_record(row_index: int, unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "retrieval_unit_id": unit["id"],
        "kind": unit["kind"],
        "source": unit["source"],
        "metadata": {
            "evidence_id": unit.get("metadata", {}).get("evidence_id"),
            "document_id": unit.get("metadata", {}).get("document_source_id")
            or unit.get("metadata", {}).get("document_id"),
            "input_artifact": unit.get("metadata", {}).get("input_artifact"),
        },
    }


def _query_index_record(row_index: int, question: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "question_id": question["question_id"],
        "intent_id": question.get("intent_id"),
        "split": question.get("split"),
        "answerability": question.get("answerability"),
        "query": question.get("query"),
    }


def _rank_query(
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
    units: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    similarities = document_matrix @ query_vector
    order = np.lexsort((np.arange(len(similarities)), -similarities))[:top_k]
    results: list[dict[str, Any]] = []
    for rank, row_index in enumerate(order, start=1):
        unit = units[int(row_index)]
        metadata = unit.get("metadata", {})
        results.append(
            {
                "rank": rank,
                "retrieval_unit_id": unit["id"],
                "kind": unit["kind"],
                "score": float(similarities[row_index]),
                "source": unit["source"],
                "evidence_id": metadata.get("evidence_id"),
                "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
            }
        )
    return results


def _dcg(binary_relevance: list[int], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(binary_relevance[:k]))


def _metrics_for_question(ranked_ids: list[str], gold_ids: list[str], k_values: tuple[int, ...]) -> dict[str, float]:
    gold = set(gold_ids)
    if not gold:
        return {}
    metrics: dict[str, float] = {}
    first_rank = next((index + 1 for index, item in enumerate(ranked_ids) if item in gold), None)
    metrics["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank
    for k in k_values:
        top_k = ranked_ids[:k]
        hits = len(set(top_k) & gold)
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        metrics[f"recall@{k}"] = hits / len(gold)
    rel = [1 if item in gold else 0 for item in ranked_ids]
    ideal = [1] * min(len(gold), 10)
    ideal_dcg = _dcg(ideal, 10)
    metrics["ndcg@10"] = 0.0 if ideal_dcg == 0 else _dcg(rel, 10) / ideal_dcg
    return metrics


def _aggregate(question_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not question_metrics:
        return {}
    keys = sorted({key for row in question_metrics for key in row})
    return {
        key: float(sum(row.get(key, 0.0) for row in question_metrics) / len(question_metrics))
        for key in keys
    }


def compute_metrics(
    questions: list[dict[str, Any]],
    results_by_question: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_question: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    by_split: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_answerability: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_task_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    abstain_questions: list[dict[str, Any]] = []
    hard_negative_diagnostics: list[dict[str, Any]] = []

    for question in questions:
        question_id = question["question_id"]
        ranked = results_by_question[question_id]
        ranked_ids = [row["retrieval_unit_id"] for row in ranked]
        gold_ids = list(question.get("gold_chunk_ids", []))
        hard_negative_ids = list(question.get("hard_negative_chunk_ids", []))
        q_metrics = _metrics_for_question(ranked_ids, gold_ids, METRIC_CUTOFFS)
        first_gold_rank = next(
            (index + 1 for index, unit_id in enumerate(ranked_ids) if unit_id in set(gold_ids)),
            None,
        )
        first_hard_rank = next(
            (
                index + 1
                for index, unit_id in enumerate(ranked_ids)
                if unit_id in set(hard_negative_ids)
            ),
            None,
        )
        per_question.append(
            {
                "question_id": question_id,
                "split": question.get("split"),
                "answerability": question.get("answerability"),
                "task_family": question.get("task_family"),
                "gold_chunk_count": len(gold_ids),
                "first_gold_rank": first_gold_rank,
                "metrics": q_metrics,
            }
        )
        if question.get("answerability") == "abstain":
            scores = [row["score"] for row in ranked[:5]]
            abstain_questions.append(
                {
                    "question_id": question_id,
                    "top1_unit": ranked[0]["retrieval_unit_id"] if ranked else None,
                    "top1_similarity": ranked[0]["score"] if ranked else None,
                    "top5_units": [row["retrieval_unit_id"] for row in ranked[:5]],
                    "top5_similarities": scores,
                }
            )
        else:
            metric_rows.append(q_metrics)
            by_split[str(question.get("split"))].append(q_metrics)
            by_answerability[str(question.get("answerability"))].append(q_metrics)
            by_task_family[str(question.get("task_family"))].append(q_metrics)

        if hard_negative_ids:
            hard_negative_diagnostics.append(
                {
                    "question_id": question_id,
                    "first_gold_rank": first_gold_rank,
                    "first_hard_negative_rank": first_hard_rank,
                    "hard_negative_outranks_first_gold": (
                        first_hard_rank is not None
                        and (first_gold_rank is None or first_hard_rank < first_gold_rank)
                    ),
                    "hard_negative_in_top5": first_hard_rank is not None and first_hard_rank <= 5,
                }
            )

    hard_with_gold = [
        row
        for row in hard_negative_diagnostics
        if row["first_gold_rank"] is not None and row["first_hard_negative_rank"] is not None
    ]
    gold_win_rate = (
        sum(row["first_gold_rank"] < row["first_hard_negative_rank"] for row in hard_with_gold)
        / len(hard_with_gold)
        if hard_with_gold
        else None
    )
    return {
        "definitions": {
            "hit@k": "1 when at least one gold chunk appears in top-k, else 0",
            "recall@k": "retrieved gold chunks in top-k divided by total gold chunks",
            "mrr": "reciprocal rank of first gold chunk",
            "ndcg@10": "binary relevance nDCG at rank 10",
            "ordinary_metric_denominator": "non-abstain questions only",
        },
        "overall_non_abstain": _aggregate(metric_rows),
        "by_split": {key: _aggregate(value) for key, value in sorted(by_split.items())},
        "by_answerability": {
            key: _aggregate(value) for key, value in sorted(by_answerability.items())
        },
        "by_task_family": {
            key: _aggregate(value) for key, value in sorted(by_task_family.items())
        },
        "per_question": per_question,
        "hard_negative_analysis": {
            "diagnostics": hard_negative_diagnostics,
            "question_count": len(hard_negative_diagnostics),
            "hard_negative_outranks_first_gold_count": sum(
                row["hard_negative_outranks_first_gold"] for row in hard_negative_diagnostics
            ),
            "hard_negative_in_top5_count": sum(
                row["hard_negative_in_top5"] for row in hard_negative_diagnostics
            ),
            "gold_vs_hard_negative_win_rate": gold_win_rate,
        },
        "abstain_questions": abstain_questions,
    }


def run_phase5a(
    repo_root: Path | None = None,
    model_name: str = MODEL_NAME,
    query_prefix: str = QUERY_PREFIX,
    top_k: int = TOP_K,
    device: str | None = None,
) -> Phase5AResult:
    """Run local vanilla dense retrieval and exact cosine evaluation."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / "data" / "processed" / "phase5a"
    output_root.mkdir(parents=True, exist_ok=True)

    units_path = repo_root / "data" / "processed" / "phase4_5" / "retrieval_units.jsonl"
    benchmark_path = repo_root / "data" / "processed" / "phase4" / "retrieval_eval.jsonl"
    units = _read_jsonl(units_path)
    questions = _read_jsonl(benchmark_path)
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("Retrieval unit IDs must be unique.")
    if len({question["question_id"] for question in questions}) != len(questions):
        raise ValueError("Question IDs must be unique.")

    model = _load_model(model_name, device=device)
    resolved_device = str(model.device)
    document_texts = [str(unit["retrieval_text"]) for unit in units]
    query_texts = [query_prefix + str(question["query"]) for question in questions]
    document_matrix = _encode_texts(model, document_texts)
    query_matrix = _encode_texts(model, query_texts)
    embedding_dimension = int(document_matrix.shape[1])
    if query_matrix.shape[1] != embedding_dimension:
        raise ValueError("Query/document embedding dimensions differ.")

    doc_embeddings_path = output_root / "document_embeddings.npy"
    query_embeddings_path = output_root / "query_embeddings.npy"
    np.save(doc_embeddings_path, document_matrix)
    np.save(query_embeddings_path, query_matrix)

    document_index = [_index_record(index, unit) for index, unit in enumerate(units)]
    query_index = [_query_index_record(index, question) for index, question in enumerate(questions)]
    document_index_path = output_root / "document_embedding_index.jsonl"
    query_index_path = output_root / "query_embedding_index.jsonl"
    _write_jsonl(document_index, document_index_path)
    _write_jsonl(query_index, query_index_path)

    result_records: list[dict[str, Any]] = []
    results_by_question: dict[str, list[dict[str, Any]]] = {}
    for query_index_value, question in enumerate(questions):
        ranked = _rank_query(query_matrix[query_index_value], document_matrix, units, top_k=top_k)
        results_by_question[question["question_id"]] = ranked
        result_records.append(
            {
                "question_id": question["question_id"],
                "query": question["query"],
                "split": question.get("split"),
                "answerability": question.get("answerability"),
                "task_family": question.get("task_family"),
                "gold_chunk_ids": question.get("gold_chunk_ids", []),
                "hard_negative_chunk_ids": question.get("hard_negative_chunk_ids", []),
                "results": ranked,
            }
        )
    dense_results_path = output_root / "dense_results.jsonl"
    _write_jsonl(result_records, dense_results_path)

    metrics = compute_metrics(questions, results_by_question)
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    output_paths = [
        doc_embeddings_path,
        document_index_path,
        query_embeddings_path,
        query_index_path,
        dense_results_path,
        metrics_path,
    ]
    output_hashes = {
        path.relative_to(repo_root).as_posix(): sha256_for_file(path) for path in output_paths
    }
    manifest = {
        "phase": "5A",
        "version": "1.0.0",
        "retrieval_units_encoded": len(units),
        "benchmark_questions_encoded": len(questions),
        "embedding_model": {
            "name": model_name,
            "revision": _model_revision(model),
            "embedding_dimension": embedding_dimension,
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "device": resolved_device,
            "dtype": str(document_matrix.dtype),
            "query_prefix": query_prefix,
        },
        "normalization": "L2 normalized; exact cosine via dot product",
        "corpus_input_hash": sha256_for_file(units_path),
        "benchmark_input_hash": sha256_for_file(benchmark_path),
        "output_artifact_hashes": output_hashes,
        "top_k": top_k,
        "metric_definitions": metrics["definitions"],
    }
    manifest_path = repo_root / "data" / "processed" / "manifests" / "phase_5a_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return Phase5AResult(
        metrics=metrics,
        manifest=manifest,
        outputs=output_paths + [manifest_path],
        manifest_path=manifest_path,
    )
