"""Phase 5B Qdrant-backed dense retrieval parity experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from materials_rag.ingestion.dense_retrieval import TOP_K, _rank_query, compute_metrics
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

COLLECTION_NAME = "materials_rag_dense_v1"
VECTOR_DIMENSION = 768
METRIC_TOLERANCE = 1e-6
LABEL_LEAKAGE_KEYS = {
    "acceptable_supporting_chunk_ids",
    "answer_key",
    "evidence_requirements",
    "forbidden_claims",
    "gold_chunk_ids",
    "gold_evidence_ids",
    "hard_negative_chunk_ids",
    "hard_negative_evidence_ids",
    "hard_negatives",
    "label_status",
    "numeric_checks",
    "required_claims",
}


@dataclass
class Phase5BResult:
    metrics: dict[str, Any]
    parity_report: dict[str, Any]
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


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _path_for_manifest(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _metadata(unit: dict[str, Any]) -> dict[str, Any]:
    return unit.get("metadata", {})


def qdrant_payload(row_index: int, unit: dict[str, Any]) -> dict[str, Any]:
    """Build a lightweight search payload without benchmark labels."""

    metadata = _metadata(unit)
    source = unit["source"]
    payload = {
        "retrieval_unit_id": unit["id"],
        "row_index": row_index,
        "kind": unit["kind"],
        "source": source,
        "source_dataset": source.get("source_dataset") or source.get("dataset"),
        "source_path": source.get("source_path"),
        "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
        "evidence_id": metadata.get("evidence_id"),
        "input_artifact": metadata.get("input_artifact"),
        "canonical_kind": metadata.get("canonical_kind"),
    }
    leaked = sorted(LABEL_LEAKAGE_KEYS & payload.keys())
    if leaked:
        raise ValueError(f"Qdrant payload contains evaluation-label keys: {leaked}")
    return payload


def _validate_artifacts(
    units: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    document_index: list[dict[str, Any]],
    query_index: list[dict[str, Any]],
    document_matrix: np.ndarray,
    query_matrix: np.ndarray,
) -> None:
    if document_matrix.shape != (len(units), VECTOR_DIMENSION):
        raise ValueError(
            f"Expected document embeddings shape ({len(units)}, {VECTOR_DIMENSION}), "
            f"got {document_matrix.shape}."
        )
    if query_matrix.shape != (len(questions), VECTOR_DIMENSION):
        raise ValueError(
            f"Expected query embeddings shape ({len(questions)}, {VECTOR_DIMENSION}), "
            f"got {query_matrix.shape}."
        )
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("Retrieval unit IDs must be unique.")
    if [row["retrieval_unit_id"] for row in document_index] != [unit["id"] for unit in units]:
        raise ValueError("Phase 5A document embedding index does not align with retrieval units.")
    if [row["question_id"] for row in query_index] != [question["question_id"] for question in questions]:
        raise ValueError("Phase 5A query embedding index does not align with benchmark questions.")


def load_phase5b_inputs(repo_root: Path) -> dict[str, Any]:
    phase5a = repo_root / "data" / "processed" / "phase5a"
    units_path = repo_root / "data" / "processed" / "phase4_5" / "retrieval_units.jsonl"
    benchmark_path = repo_root / "data" / "processed" / "phase4" / "retrieval_eval.jsonl"
    document_index_path = phase5a / "document_embedding_index.jsonl"
    query_index_path = phase5a / "query_embedding_index.jsonl"
    document_embeddings_path = phase5a / "document_embeddings.npy"
    query_embeddings_path = phase5a / "query_embeddings.npy"

    units = _read_jsonl(units_path)
    questions = _read_jsonl(benchmark_path)
    document_index = _read_jsonl(document_index_path)
    query_index = _read_jsonl(query_index_path)
    document_matrix = np.load(document_embeddings_path)
    query_matrix = np.load(query_embeddings_path)
    _validate_artifacts(units, questions, document_index, query_index, document_matrix, query_matrix)

    return {
        "units_path": units_path,
        "benchmark_path": benchmark_path,
        "document_index_path": document_index_path,
        "query_index_path": query_index_path,
        "document_embeddings_path": document_embeddings_path,
        "query_embeddings_path": query_embeddings_path,
        "units": units,
        "questions": questions,
        "document_index": document_index,
        "query_index": query_index,
        "document_matrix": document_matrix,
        "query_matrix": query_matrix,
    }


def open_qdrant(storage_path: Path) -> QdrantClient:
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def rebuild_collection(
    client: QdrantClient,
    collection_name: str,
    units: list[dict[str, Any]],
    document_matrix: np.ndarray,
) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=row_index,
            vector=document_matrix[row_index].astype(float).tolist(),
            payload=qdrant_payload(row_index, unit),
        )
        for row_index, unit in enumerate(units)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)


def qdrant_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    limit: int = TOP_K,
) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.astype(float).tolist(),
        limit=limit,
        with_payload=True,
    )
    results: list[dict[str, Any]] = []
    for rank, point in enumerate(response.points, start=1):
        payload = dict(point.payload or {})
        results.append(
            {
                "rank": rank,
                "retrieval_unit_id": payload["retrieval_unit_id"],
                "kind": payload["kind"],
                "score": float(point.score),
                "source": payload["source"],
                "evidence_id": payload.get("evidence_id"),
                "document_id": payload.get("document_id"),
            }
        )
    return results


def _spearman_rank_correlation(exact: list[dict[str, Any]], qdrant: list[dict[str, Any]]) -> float:
    exact_ranks = {row["retrieval_unit_id"]: row["rank"] for row in exact}
    qdrant_ranks = {row["retrieval_unit_id"]: row["rank"] for row in qdrant}
    ids = sorted(set(exact_ranks) | set(qdrant_ranks))
    if len(ids) < 2:
        return 1.0
    missing_rank = max(len(exact), len(qdrant)) + 1
    deltas = [
        (exact_ranks.get(unit_id, missing_rank) - qdrant_ranks.get(unit_id, missing_rank)) ** 2
        for unit_id in ids
    ]
    n = len(ids)
    return float(1.0 - (6.0 * sum(deltas)) / (n * (n**2 - 1)))


def compare_with_exact(
    questions: list[dict[str, Any]],
    exact_by_question: dict[str, list[dict[str, Any]]],
    qdrant_by_question: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    top1_matches = 0
    top5_set_matches = 0
    top10_set_matches = 0
    score_differences: list[float] = []
    correlations: list[float] = []

    for question in questions:
        question_id = question["question_id"]
        exact = exact_by_question[question_id]
        qdrant = qdrant_by_question[question_id]
        exact_ids = [row["retrieval_unit_id"] for row in exact]
        qdrant_ids = [row["retrieval_unit_id"] for row in qdrant]
        if exact_ids[:1] == qdrant_ids[:1]:
            top1_matches += 1
        if set(exact_ids[:5]) == set(qdrant_ids[:5]):
            top5_set_matches += 1
        if set(exact_ids[:10]) == set(qdrant_ids[:10]):
            top10_set_matches += 1
        qdrant_scores = {row["retrieval_unit_id"]: row["score"] for row in qdrant}
        for row in exact:
            if row["retrieval_unit_id"] in qdrant_scores:
                score_differences.append(abs(row["score"] - qdrant_scores[row["retrieval_unit_id"]]))
        correlations.append(_spearman_rank_correlation(exact, qdrant))
        if exact_ids != qdrant_ids:
            first_mismatch = next(
                index
                for index, (exact_id, qdrant_id) in enumerate(zip(exact_ids, qdrant_ids), start=1)
                if exact_id != qdrant_id
            )
            differences.append(
                {
                    "question_id": question_id,
                    "first_mismatch_rank": first_mismatch,
                    "exact_top20": exact_ids,
                    "qdrant_top20": qdrant_ids,
                }
            )

    total = len(questions)
    return {
        "comparison": "Qdrant local persistent cosine search against exact NumPy dot-product rankings",
        "qdrant_search_params": {
            "local_persistent_mode": True,
            "local_mode_search": "exact brute-force",
            "metadata_filters": False,
        },
        "top1_agreement": top1_matches / total,
        "top5_set_agreement": top5_set_matches / total,
        "top10_set_agreement": top10_set_matches / total,
        "mean_spearman_top20": float(sum(correlations) / len(correlations)),
        "minimum_spearman_top20": float(min(correlations)),
        "maximum_score_difference": float(max(score_differences) if score_differences else 0.0),
        "ranking_difference_count": len(differences),
        "ranking_differences": differences,
    }


def run_phase5b(
    repo_root: Path | None = None,
    storage_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
) -> Phase5BResult:
    """Load Phase 5A embeddings into local Qdrant and evaluate ranking parity."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if storage_path is None:
        storage_path = repo_root / "data" / "processed" / "phase5b" / "qdrant_storage"

    inputs = load_phase5b_inputs(repo_root)
    units = inputs["units"]
    questions = inputs["questions"]
    document_matrix = inputs["document_matrix"]
    query_matrix = inputs["query_matrix"]

    client = open_qdrant(storage_path)
    try:
        rebuild_collection(client, collection_name, units, document_matrix)
        collection_count = client.count(collection_name=collection_name, exact=True).count

        result_records: list[dict[str, Any]] = []
        qdrant_by_question: dict[str, list[dict[str, Any]]] = {}
        exact_by_question: dict[str, list[dict[str, Any]]] = {}
        for query_row, question in enumerate(questions):
            question_id = question["question_id"]
            qdrant_results = qdrant_search(client, collection_name, query_matrix[query_row], top_k)
            exact_results = _rank_query(query_matrix[query_row], document_matrix, units, top_k=top_k)
            qdrant_by_question[question_id] = qdrant_results
            exact_by_question[question_id] = exact_results
            result_records.append(
                {
                    "question_id": question_id,
                    "query_embedding_row_index": query_row,
                    "results": qdrant_results,
                }
            )
    finally:
        client.close()

    metrics = compute_metrics(questions, qdrant_by_question)
    parity_report = compare_with_exact(questions, exact_by_question, qdrant_by_question)

    output_root = repo_root / "data" / "processed" / "phase5b"
    results_path = output_root / "qdrant_results.jsonl"
    metrics_path = output_root / "metrics.json"
    parity_path = output_root / "parity_report.json"
    _write_jsonl(result_records, results_path)
    _write_json(metrics, metrics_path)
    _write_json(parity_report, parity_path)

    output_paths = [results_path, metrics_path, parity_path]
    manifest_path = repo_root / "data" / "processed" / "manifests" / "phase_5b_manifest.json"
    manifest = {
        "phase": "5B",
        "version": "1.0.0",
        "objective": "Qdrant-backed dense retrieval parity with Phase 5A exact cosine search",
        "collection_name": collection_name,
        "storage_path": _path_for_manifest(storage_path, repo_root),
        "retrieval_units_indexed": collection_count,
        "benchmark_questions_searched": len(questions),
        "embedding_source": "Phase 5A persisted BGE embeddings; no corpus or query re-embedding",
        "embedding_dimension": VECTOR_DIMENSION,
        "distance": "cosine",
        "search_params": {
            "local_persistent_mode": True,
            "local_mode_search": "exact brute-force",
            "metadata_filters": False,
            "top_k": top_k,
        },
        "input_hashes": {
            path.relative_to(repo_root).as_posix(): sha256_for_file(path)
            for path in [
                inputs["units_path"],
                inputs["benchmark_path"],
                inputs["document_index_path"],
                inputs["query_index_path"],
                inputs["document_embeddings_path"],
                inputs["query_embeddings_path"],
            ]
        },
        "output_artifact_hashes": {
            path.relative_to(repo_root).as_posix(): sha256_for_file(path) for path in output_paths
        },
        "parity_summary": {
            key: parity_report[key]
            for key in [
                "top1_agreement",
                "top5_set_agreement",
                "top10_set_agreement",
                "maximum_score_difference",
                "ranking_difference_count",
            ]
        },
        "metric_definitions": metrics["definitions"],
        "payload_policy": {
            "stores_gold_or_eval_labels": False,
            "stores_answer_keys": False,
            "stores_retrieval_text": False,
        },
    }
    _write_json(manifest, manifest_path)

    return Phase5BResult(
        metrics=metrics,
        parity_report=parity_report,
        manifest=manifest,
        outputs=output_paths + [manifest_path],
        manifest_path=manifest_path,
    )
