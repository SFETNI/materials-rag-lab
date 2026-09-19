"""Phase 8B controlled query decomposition retrieval infrastructure and evaluation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from materials_rag.ingestion.dense_retrieval import MODEL_NAME as BI_ENCODER_MODEL_NAME
from materials_rag.ingestion.dense_retrieval import QUERY_PREFIX, _encode_texts, compute_metrics
from materials_rag.ingestion.generation_packets import PROMPT_PATH as GENERATOR_PROMPT_PATH
from materials_rag.ingestion.generation_packets import (
    SYSTEM_PROMPT_VERSION,
    _context_item,
    _packet_hash,
    _sha256_text,
    format_generator_input,
    validate_packets,
)
from materials_rag.ingestion.generation_packets import TOP_K as GENERATION_TOP_K
from materials_rag.ingestion.hybrid_retrieval import (
    BM25_TOP_K,
    DENSE_TOP_K,
    RRF_K,
    bm25_search,
    build_bm25_index,
    fuse_dense_bm25,
)
from materials_rag.ingestion.multi_query_retrieval import repair_text_encoding
from materials_rag.ingestion.public_sanity import PUBLIC_SANITY_QUESTIONS
from materials_rag.ingestion.qdrant_retrieval import COLLECTION_NAME, open_qdrant, qdrant_search
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

DECOMPOSER_PROMPT_VERSION = "query_decomposer_v1"
DECOMPOSER_PROMPT_PATH = Path("docs/generation/query_decomposer_v1.md")
PHASE8B_ROOT = Path("data/processed/phase8b")
DECOMPOSITION_PACKETS_PATH = PHASE8B_ROOT / "decomposition_packets.jsonl"
PUBLIC_DECOMPOSITION_PACKETS_PATH = PHASE8B_ROOT / "public_sanity_decomposition_packets.jsonl"
QUERY_DECOMPOSITIONS_PATH = PHASE8B_ROOT / "query_decompositions.jsonl"
PUBLIC_DECOMPOSITIONS_PATH = PHASE8B_ROOT / "public_sanity_query_decompositions.jsonl"
RAW_QUERY_DECOMPOSITIONS_PATH = PHASE8B_ROOT / "raw/query_decompositions_raw.jsonl"
RAW_PUBLIC_DECOMPOSITIONS_PATH = PHASE8B_ROOT / "raw/public_sanity_query_decompositions_raw.jsonl"
ENCODING_NORMALIZATION_REPORT_PATH = PHASE8B_ROOT / "encoding_normalization_report.json"
BATCH_PROMPT_PATH = Path("experiments/18_query_decomposer_batch_prompt.txt")
HYBRID_CANDIDATE_K = 50
DECOMPOSITION_RRF_K = 60
CRITICAL_FAILURE_QUESTION_IDS = ["SYNQ-002-A", "SYNQ-009-B", "SYNQ-014-B"]
INSPECTION_QUESTION_IDS = ["SYNQ-001-A", "SYNQ-012-A", *CRITICAL_FAILURE_QUESTION_IDS]
TRACKED_EVIDENCE_IDS = {
    "SYNQ-001-A": ["FAT-B017#results", "FAT-B017#conditions"],
    "SYNQ-012-A": ["RCA-B017-004-rev2#status", "RCA-B017-004-rev2#conclusion"],
}
LEAKAGE_PACKET_KEYS = {
    "gold_chunk_ids",
    "gold_evidence_ids",
    "hard_negative_chunk_ids",
    "hard_negative_evidence_ids",
    "answerability",
    "split",
    "task_family",
    "reference_answer",
    "answer_key",
    "evidence_requirements",
    "required_claims",
    "forbidden_claims",
    "numeric_checks",
    "retrieval_unit_ids",
    "scores",
    "ranks",
}


class DecompositionImportError(ValueError):
    """Raised when external isolated decomposition output fails validation."""


class DecompositionFilesMissingError(FileNotFoundError):
    """Raised when Phase 8B decomposition files have not yet been provided."""


@dataclass
class Phase8BResult:
    status: str
    packets: list[dict[str, Any]]
    public_packets: list[dict[str, Any]]
    outputs: list[Path]
    metrics: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    missing_decomposition_files: list[Path] | None = None


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


def _safe_packet(question_id: str, question: str) -> dict[str, str]:
    return {"question_id": question_id, "original_question": question}


def build_decomposition_packets(repo_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    questions = _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl")
    synthetic_packets = [_safe_packet(row["question_id"], row["query"]) for row in questions]
    public_packets = [_safe_packet(row["question_id"], row["query"]) for row in PUBLIC_SANITY_QUESTIONS]
    validate_safe_decomposition_packets(synthetic_packets)
    validate_safe_decomposition_packets(public_packets)
    return synthetic_packets, public_packets


def validate_safe_decomposition_packets(packets: list[dict[str, Any]]) -> None:
    for packet in packets:
        if set(packet) != {"question_id", "original_question"}:
            raise ValueError(f"Unsafe decomposition packet keys: {sorted(packet)}")
        if not packet["question_id"] or not packet["original_question"]:
            raise ValueError("Decomposition packets require non-empty question_id and original_question.")
        if LEAKAGE_PACKET_KEYS & set(packet):
            raise ValueError(
                f"Decomposition packet contains leakage keys: {LEAKAGE_PACKET_KEYS & set(packet)}"
            )


def write_decomposer_batch_prompt(
    repo_root: Path,
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> Path:
    prompt_text = (repo_root / DECOMPOSER_PROMPT_PATH).read_text(encoding="utf-8").strip()
    lines = [
        prompt_text,
        "",
        "Decompose the following question-only requests. Return JSONL only, preserving input order.",
        "One JSON object per input line. No Markdown fences. No explanations.",
        "Do not use tools, web, repository files, retrieval results, gold labels, or benchmark metadata.",
        "",
        "SYNTHETIC_REQUESTS_JSONL:",
    ]
    lines.extend(json.dumps(packet, ensure_ascii=False, sort_keys=True) for packet in synthetic_packets)
    lines.extend(["", "PUBLIC_SANITY_REQUESTS_JSONL:"])
    lines.extend(json.dumps(packet, ensure_ascii=False, sort_keys=True) for packet in public_packets)
    path = repo_root / BATCH_PROMPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _preserve_raw_file(source: Path, raw_path: Path) -> None:
    if not source.exists():
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        shutil.copy2(source, raw_path)


def _normalize_decomposition_records(
    raw_path: Path,
    output_path: Path,
    expected_packets: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _read_jsonl(raw_path)
    expected_by_id = {packet["question_id"]: packet["original_question"] for packet in expected_packets}
    normalized_records: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for record in records:
        question_id = record.get("question_id")
        expected_question = expected_by_id.get(str(question_id))
        new_record = dict(record)
        if isinstance(new_record.get("original_question"), str):
            before = new_record["original_question"]
            after, method = repair_text_encoding(before, expected_question)
            if method:
                new_record["original_question"] = after
                changes.append(
                    {
                        "question_id": question_id,
                        "field": "original_question",
                        "before": before,
                        "after": after,
                        "repair_method": method,
                    }
                )
        if isinstance(new_record.get("subqueries"), list):
            repaired_subqueries = []
            for index, subquery in enumerate(new_record["subqueries"]):
                if isinstance(subquery, str):
                    after, method = repair_text_encoding(subquery)
                    if method:
                        changes.append(
                            {
                                "question_id": question_id,
                                "field": f"subqueries[{index}]",
                                "before": subquery,
                                "after": after,
                                "repair_method": method,
                            }
                        )
                    repaired_subqueries.append(after)
                else:
                    repaired_subqueries.append(subquery)
            new_record["subqueries"] = repaired_subqueries
        normalized_records.append(new_record)
    _write_jsonl(output_path, normalized_records)
    return normalized_records, changes


def normalize_phase8b_decomposition_inputs(
    repo_root: Path,
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> dict[str, Any]:
    query_path = repo_root / QUERY_DECOMPOSITIONS_PATH
    public_path = repo_root / PUBLIC_DECOMPOSITIONS_PATH
    raw_query_path = repo_root / RAW_QUERY_DECOMPOSITIONS_PATH
    raw_public_path = repo_root / RAW_PUBLIC_DECOMPOSITIONS_PATH
    _preserve_raw_file(query_path, raw_query_path)
    _preserve_raw_file(public_path, raw_public_path)
    changes: list[dict[str, Any]] = []
    if raw_query_path.exists():
        _, synthetic_changes = _normalize_decomposition_records(
            raw_query_path, query_path, synthetic_packets
        )
        changes.extend(synthetic_changes)
    if raw_public_path.exists():
        _, public_changes = _normalize_decomposition_records(
            raw_public_path, public_path, public_packets
        )
        changes.extend(public_changes)
    raw_paths = [path for path in [raw_query_path, raw_public_path] if path.exists()]
    normalized_paths = [path for path in [query_path, public_path] if path.exists()]
    report = {
        "phase": "8B",
        "normalization": "deterministic transport encoding repair only",
        "raw_file_hashes": {_relative(path, repo_root): sha256_for_file(path) for path in raw_paths},
        "normalized_file_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in normalized_paths
        },
        "change_count": len(changes),
        "changes": changes,
    }
    _write_json(repo_root / ENCODING_NORMALIZATION_REPORT_PATH, report)
    return report


def validate_decompositions(
    decomposition_path: Path,
    expected_packets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not decomposition_path.exists():
        raise DecompositionFilesMissingError(f"Missing external decomposition file: {decomposition_path}")
    records = _read_jsonl(decomposition_path)
    expected_ids = [packet["question_id"] for packet in expected_packets]
    actual_ids = [record.get("question_id") for record in records]
    if len(actual_ids) != len(set(actual_ids)):
        duplicates = sorted({question_id for question_id in actual_ids if actual_ids.count(question_id) > 1})
        raise DecompositionImportError(f"Duplicate decomposition question IDs: {duplicates}")
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise DecompositionImportError(
            "Decomposition IDs must match expected packet order exactly. "
            f"missing={missing}; unexpected={unexpected}."
        )
    expected_by_id = {packet["question_id"]: packet for packet in expected_packets}
    validated: list[dict[str, Any]] = []
    for record in records:
        if set(record) != {"question_id", "original_question", "decomposable", "subqueries"}:
            raise DecompositionImportError(
                f"{record.get('question_id')} has invalid keys: {sorted(record)}"
            )
        question_id = record["question_id"]
        expected_question = expected_by_id[question_id]["original_question"]
        if record["original_question"] != expected_question:
            raise DecompositionImportError(
                f"{question_id} original_question does not match frozen text."
            )
        decomposable = record["decomposable"]
        if not isinstance(decomposable, bool):
            raise DecompositionImportError(f"{question_id} decomposable must be boolean.")
        subqueries = record["subqueries"]
        if not isinstance(subqueries, list):
            raise DecompositionImportError(f"{question_id} subqueries must be a list.")
        normalized_subqueries: list[str] = []
        for subquery in subqueries:
            if not isinstance(subquery, str) or not subquery.strip():
                raise DecompositionImportError(
                    f"{question_id} contains an empty or non-string subquery."
                )
            text = subquery.strip()
            if text == expected_question:
                raise DecompositionImportError(
                    f"{question_id} subquery is identical to the original question."
                )
            normalized_subqueries.append(text)
        if decomposable and not (2 <= len(normalized_subqueries) <= 4):
            raise DecompositionImportError(
                f"{question_id} decomposable=true requires 2 to 4 subqueries."
            )
        if not decomposable and normalized_subqueries:
            raise DecompositionImportError(
                f"{question_id} decomposable=false requires zero subqueries."
            )
        if len(set(normalized_subqueries)) != len(normalized_subqueries):
            raise DecompositionImportError(f"{question_id} subqueries must be unique.")
        validated.append(
            {
                "question_id": question_id,
                "original_question": expected_question,
                "decomposable": decomposable,
                "subqueries": normalized_subqueries,
            }
        )
    return validated


def _query_variants(record: dict[str, Any]) -> list[dict[str, str]]:
    variants = [{"variant": "original", "query": record["original_question"]}]
    if record["decomposable"]:
        variants.extend(
            {"variant": f"subquery_{index}", "query": subquery}
            for index, subquery in enumerate(record["subqueries"], start=1)
        )
    return variants


def decomposition_rrf_score(
    query_ranks: dict[str, int | None], rrf_k: int = DECOMPOSITION_RRF_K
) -> float:
    return float(sum(1.0 / (rrf_k + rank) for rank in query_ranks.values() if rank is not None))


def _dense_for_variant_queries(
    repo_root: Path, variant_rows: list[dict[str, str]]
) -> dict[str, list[dict[str, Any]]]:
    model = SentenceTransformer(BI_ENCODER_MODEL_NAME, local_files_only=True)
    model.eval()
    query_matrix = _encode_texts(model, [QUERY_PREFIX + row["query"] for row in variant_rows])
    client = open_qdrant(repo_root / "data/processed/phase5b/qdrant_storage")
    try:
        return {
            row["variant_key"]: qdrant_search(client, COLLECTION_NAME, query_matrix[index], DENSE_TOP_K)
            for index, row in enumerate(variant_rows)
        }
    finally:
        client.close()


def _hybrid_rankings_for_decompositions(
    repo_root: Path,
    decomposition_records: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units_by_id = {unit["id"]: unit for unit in units}
    bm25_index = build_bm25_index(units)
    variant_rows = []
    for record in decomposition_records:
        for variant in _query_variants(record):
            variant_rows.append(
                {
                    "question_id": record["question_id"],
                    "variant": variant["variant"],
                    "variant_key": f"{record['question_id']}::{variant['variant']}",
                    "query": variant["query"],
                }
            )
    dense_by_variant = _dense_for_variant_queries(repo_root, variant_rows)
    hybrid_by_variant: dict[str, dict[str, Any]] = {}
    for row in variant_rows:
        bm25 = bm25_search(bm25_index, row["query"], BM25_TOP_K)
        hybrid_by_variant[row["variant_key"]] = fuse_dense_bm25(
            {"question_id": row["variant_key"], "query": row["query"]},
            dense_by_variant[row["variant_key"]],
            bm25,
            units_by_id,
            RRF_K,
        )
    return [_fuse_decomposition(record, hybrid_by_variant) for record in decomposition_records]


def _fuse_decomposition(
    decomposition_record: dict[str, Any],
    hybrid_by_variant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variants = [variant["variant"] for variant in _query_variants(decomposition_record)]
    variant_results = {
        variant: hybrid_by_variant[f"{decomposition_record['question_id']}::{variant}"]["results"][
            :HYBRID_CANDIDATE_K
        ]
        for variant in variants
    }
    candidate_ids = sorted(
        {result["retrieval_unit_id"] for results in variant_results.values() for result in results}
    )
    by_variant_id = {
        variant: {row["retrieval_unit_id"]: row for row in results}
        for variant, results in variant_results.items()
    }
    fused = []
    for unit_id in candidate_ids:
        query_ranks = {
            variant: by_variant_id[variant].get(unit_id, {}).get("hybrid_rank")
            for variant in variants
        }
        exemplar = next(
            by_variant_id[variant][unit_id] for variant in variants if unit_id in by_variant_id[variant]
        )
        row = {
            "retrieval_unit_id": unit_id,
            "decomposition_rrf_score": decomposition_rrf_score(query_ranks),
            "original_hybrid_rank": query_ranks.get("original"),
            "query_hits": [
                {
                    "variant": variant,
                    "hybrid_rank": query_ranks[variant],
                    "rrf_score": by_variant_id[variant].get(unit_id, {}).get("rrf_score"),
                }
                for variant in variants
                if unit_id in by_variant_id[variant]
            ],
            "kind": exemplar["kind"],
            "source": exemplar["source"],
            "evidence_id": exemplar.get("evidence_id"),
            "document_id": exemplar.get("document_id"),
        }
        for variant in variants:
            if variant != "original":
                row[f"{variant}_hybrid_rank"] = query_ranks.get(variant)
        fused.append(row)
    fused.sort(
        key=lambda row: (
            -row["decomposition_rrf_score"],
            row["original_hybrid_rank"] if row["original_hybrid_rank"] is not None else 999,
            row.get("subquery_1_hybrid_rank") if row.get("subquery_1_hybrid_rank") is not None else 999,
            row["retrieval_unit_id"],
        )
    )
    for rank, row in enumerate(fused, start=1):
        row["decomposition_rank"] = rank
    return {
        "question_id": decomposition_record["question_id"],
        "query": decomposition_record["original_question"],
        "decomposable": decomposition_record["decomposable"],
        "subqueries": decomposition_record["subqueries"],
        "retrieval_query_count": len(variants),
        "hybrid_candidate_k_per_query": HYBRID_CANDIDATE_K,
        "results": fused,
    }


def _ranked_for_metrics(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["question_id"]: [
            {"retrieval_unit_id": result["retrieval_unit_id"], "score": result["decomposition_rrf_score"]}
            for result in sorted(row["results"], key=lambda result: result["decomposition_rank"])
        ]
        for row in records
    }


def _gold_recall_at_k(records: list[dict[str, Any]], questions: list[dict[str, Any]], k: int) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    recalls = []
    per_question = []
    for question in questions:
        gold_ids = list(question.get("gold_chunk_ids", []))
        ranked_ids = [
            row["retrieval_unit_id"] for row in by_question[question["question_id"]]["results"][:k]
        ]
        ranked_set = set(ranked_ids)
        hits = [unit_id for unit_id in gold_ids if unit_id in ranked_set]
        missing = [unit_id for unit_id in gold_ids if unit_id not in ranked_set]
        recall = len(hits) / len(gold_ids) if gold_ids else None
        if recall is not None:
            recalls.append(recall)
        per_question.append(
            {
                "question_id": question["question_id"],
                "gold_count": len(gold_ids),
                "hits": hits,
                "missing": missing,
                "recall": recall,
            }
        )
    return {
        "k": k,
        "mean_recall_all_questions_with_gold": float(sum(recalls) / len(recalls)),
        "per_question": per_question,
    }


def _candidate_union_gold_recall(
    records: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    recalls = []
    per_question = []
    for question in questions:
        result_row = by_question[question["question_id"]]
        by_unit = {row["retrieval_unit_id"]: row for row in result_row["results"]}
        gold_ids = list(question.get("gold_chunk_ids", []))
        hits = [unit_id for unit_id in gold_ids if unit_id in by_unit]
        missing = [unit_id for unit_id in gold_ids if unit_id not in by_unit]
        recall = len(hits) / len(gold_ids) if gold_ids else None
        if recall is not None:
            recalls.append(recall)
        per_question.append(
            {
                "question_id": question["question_id"],
                "decomposable": result_row["decomposable"],
                "gold_count": len(gold_ids),
                "hits": hits,
                "missing": missing,
                "discovered_by": {
                    unit_id: [hit["variant"] for hit in by_unit[unit_id]["query_hits"]]
                    for unit_id in hits
                },
                "recall": recall,
            }
        )
    return {
        "candidate_pool": "union of original and subquery Hybrid Top-50 candidates",
        "mean_recall_all_questions_with_gold": float(sum(recalls) / len(recalls)),
        "per_question": per_question,
    }


def _comparison_table(
    phase7b_metrics: dict[str, Any], phase8a_metrics: dict[str, Any], decomp_metrics: dict[str, Any]
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
    mq_by_metric = {row["metric"]: row["multi_query_hybrid"] for row in phase8a_metrics["comparison_table"]}
    return [
        {
            "metric": metric,
            "dense": phase7b_metrics["dense"]["overall_non_abstain"][metric],
            "dense_cross_encoder": phase7b_metrics["dense_cross_encoder"]["overall_non_abstain"][metric],
            "hybrid_rrf": phase7b_metrics["hybrid_rrf"]["overall_non_abstain"][metric],
            "hybrid_rrf_cross_encoder": phase7b_metrics["hybrid_rrf_cross_encoder"][
                "overall_non_abstain"
            ][metric],
            "multi_query_hybrid": mq_by_metric[metric],
            "decomposition_hybrid": decomp_metrics["overall_non_abstain"][metric],
        }
        for metric in ordered
    ]


def _decomposition_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    decomposable = [row for row in records if row["decomposable"]]
    atomic = [row for row in records if not row["decomposable"]]
    subquery_counts = [len(row["subqueries"]) for row in decomposable]
    retrieval_counts = [row["retrieval_query_count"] for row in records]
    return {
        "question_count": len(records),
        "decomposable_count": len(decomposable),
        "atomic_count": len(atomic),
        "percent_decomposable": len(decomposable) / len(records) if records else 0.0,
        "mean_subqueries_for_decomposable": float(sum(subquery_counts) / len(subquery_counts))
        if subquery_counts
        else 0.0,
        "total_retrieval_queries_executed": sum(retrieval_counts),
        "mean_retrieval_queries_per_original_question": float(sum(retrieval_counts) / len(records))
        if records
        else 0.0,
    }


def _rank_for_evidence(row: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    for result in row["results"]:
        if result.get("evidence_id") == evidence_id:
            payload = {
                "retrieval_unit_id": result["retrieval_unit_id"],
                "decomposition_rank": result["decomposition_rank"],
                "decomposition_rrf_score": result["decomposition_rrf_score"],
                "original_hybrid_rank": result["original_hybrid_rank"],
                "query_hits": result["query_hits"],
            }
            for key, value in result.items():
                if key.startswith("subquery_") and key.endswith("_hybrid_rank"):
                    payload[key] = value
            return payload
    return None


def _inspection_report(
    questions: list[dict[str, Any]], records: list[dict[str, Any]], question_ids: list[str]
) -> dict[str, Any]:
    questions_by_id = {row["question_id"]: row for row in questions}
    records_by_id = {row["question_id"]: row for row in records}
    report = {}
    for question_id in question_ids:
        question = questions_by_id[question_id]
        row = records_by_id[question_id]
        tracked_evidence = TRACKED_EVIDENCE_IDS.get(question_id)
        if tracked_evidence is not None:
            evidence = {evidence_id: _rank_for_evidence(row, evidence_id) for evidence_id in tracked_evidence}
        else:
            by_unit = {result["retrieval_unit_id"]: result for result in row["results"]}
            evidence = {
                unit_id: {
                    "retrieval_unit_id": unit_id,
                    "decomposition_rank": by_unit.get(unit_id, {}).get("decomposition_rank"),
                    "original_hybrid_rank": by_unit.get(unit_id, {}).get("original_hybrid_rank"),
                    "query_hits": by_unit.get(unit_id, {}).get("query_hits", []),
                    **{
                        key: value
                        for key, value in by_unit.get(unit_id, {}).items()
                        if key.startswith("subquery_") and key.endswith("_hybrid_rank")
                    },
                }
                for unit_id in question.get("gold_chunk_ids", [])
            }
        report[question_id] = {
            "question": question["query"],
            "decomposable": row["decomposable"],
            "subqueries": row["subqueries"],
            "tracked_evidence": evidence,
        }
    return report


def _packet_for_generation(
    row: dict[str, Any], units_by_id: dict[str, dict[str, Any]], prompt_hash: str
) -> dict[str, Any]:
    ordered = sorted(row["results"], key=lambda result: result["decomposition_rank"])[
        :GENERATION_TOP_K
    ]
    retrieval_unit_ids = [result["retrieval_unit_id"] for result in ordered]
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in retrieval_unit_ids]
    generator_input = format_generator_input(row["query"], context_items)
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "top_k": GENERATION_TOP_K,
        "retrieval_unit_ids": retrieval_unit_ids,
        "retrieval_backend": "Phase 8B Decomposition Hybrid RRF",
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def _generation_packets(
    records: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]], prompt_hash: str
) -> list[dict[str, Any]]:
    packets = [_packet_for_generation(row, units_by_id, prompt_hash) for row in records]
    validation = validate_packets(packets, units_by_id)
    if not validation["passed"]:
        raise ValueError(f"Phase 8B generation packet validation failed: {validation}")
    for packet in packets:
        lowered = packet["generator_input"].lower()
        for term in [
            "subquery_",
            "decomposable",
            "decomposition_rrf_score",
            "hybrid_rank",
            "score:",
        ]:
            if term in lowered:
                raise ValueError(f"Phase 8B packet exposes retrieval internals: {term}")
    return packets


def _public_sanity_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    rows = []
    for question in PUBLIC_SANITY_QUESTIONS:
        result_row = by_question[question["question_id"]]
        expected = set(question["expected_retrieval_unit_ids"])
        ranked = sorted(result_row["results"], key=lambda row: row["decomposition_rank"])
        expected_rows = [row for row in ranked if row["retrieval_unit_id"] in expected]
        best = expected_rows[0] if expected_rows else None
        rows.append(
            {
                "question_id": question["question_id"],
                "query": question["query"],
                "decomposable": result_row["decomposable"],
                "subqueries": result_row["subqueries"],
                "expected_retrieval_unit_ids": question["expected_retrieval_unit_ids"],
                "best_expected_decomposition_rank": best.get("decomposition_rank") if best else None,
                "best_expected_query_hits": best.get("query_hits") if best else [],
                "decomposition_top5_ids": [row["retrieval_unit_id"] for row in ranked[:5]],
            }
        )
    return {
        "question_count": len(rows),
        "pubsan_005": next(row for row in rows if row["question_id"] == "PUBSAN-005"),
        "per_question": rows,
    }


def _write_waiting_manifest(
    repo_root: Path,
    setup_outputs: list[Path],
    missing: list[Path],
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / "data/processed/manifests/phase_8b_manifest.json"
    manifest = {
        "phase": "8B",
        "version": "1.0.0",
        "status": "waiting_for_external_decompositions",
        "objective": "Controlled query decomposition Hybrid RRF retrieval infrastructure",
        "configuration": {
            "per_query_dense_top_k": DENSE_TOP_K,
            "per_query_bm25_top_k": BM25_TOP_K,
            "per_query_hybrid_candidate_k": HYBRID_CANDIDATE_K,
            "second_stage_rrf_k": DECOMPOSITION_RRF_K,
            "cross_encoder_used": False,
        },
        "packet_counts": {"synthetic": len(synthetic_packets), "public_sanity": len(public_packets)},
        "missing_decomposition_files": [_relative(path, repo_root) for path in missing],
        "decomposer_prompt": {
            "version": DECOMPOSER_PROMPT_VERSION,
            "path": DECOMPOSER_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / DECOMPOSER_PROMPT_PATH),
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in setup_outputs
        },
    }
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def run_phase8b(repo_root: Path | None = None) -> Phase8BResult:
    """Prepare safe decomposition inputs and run Phase 8B when external files exist."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / PHASE8B_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    synthetic_packets, public_packets = build_decomposition_packets(repo_root)
    packets_path = repo_root / DECOMPOSITION_PACKETS_PATH
    public_packets_path = repo_root / PUBLIC_DECOMPOSITION_PACKETS_PATH
    _write_jsonl(packets_path, synthetic_packets)
    _write_jsonl(public_packets_path, public_packets)
    batch_prompt_path = write_decomposer_batch_prompt(repo_root, synthetic_packets, public_packets)

    decomposition_paths = [repo_root / QUERY_DECOMPOSITIONS_PATH, repo_root / PUBLIC_DECOMPOSITIONS_PATH]
    missing = [path for path in decomposition_paths if not path.exists()]
    setup_outputs = [packets_path, public_packets_path, batch_prompt_path]
    if missing:
        manifest, manifest_path = _write_waiting_manifest(
            repo_root, setup_outputs, missing, synthetic_packets, public_packets
        )
        return Phase8BResult(
            status="waiting_for_external_decompositions",
            packets=synthetic_packets,
            public_packets=public_packets,
            outputs=setup_outputs,
            manifest=manifest,
            manifest_path=manifest_path,
            missing_decomposition_files=missing,
        )

    normalization_report = normalize_phase8b_decomposition_inputs(
        repo_root, synthetic_packets, public_packets
    )
    decompositions = validate_decompositions(repo_root / QUERY_DECOMPOSITIONS_PATH, synthetic_packets)
    public_decompositions = validate_decompositions(repo_root / PUBLIC_DECOMPOSITIONS_PATH, public_packets)

    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    phase7b_metrics_path = repo_root / "data/processed/phase7b/metrics.json"
    phase8a_metrics_path = repo_root / "data/processed/phase8a/metrics.json"
    generator_prompt_path = repo_root / GENERATOR_PROMPT_PATH
    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    units_by_id = {unit["id"]: unit for unit in units}
    phase7b_metrics = _read_json(phase7b_metrics_path)
    phase8a_metrics = _read_json(phase8a_metrics_path)

    decomp_records = _hybrid_rankings_for_decompositions(repo_root, decompositions, units)
    public_decomp_records = _hybrid_rankings_for_decompositions(repo_root, public_decompositions, units)
    decomp_metrics = compute_metrics(questions, _ranked_for_metrics(decomp_records))
    generator_prompt_hash = _sha256_text(generator_prompt_path.read_text(encoding="utf-8"))
    generation_packets = _generation_packets(decomp_records, units_by_id, generator_prompt_hash)
    public_report = _public_sanity_report(public_decomp_records)
    diagnostics = _decomposition_diagnostics(decomp_records)

    metrics_payload = {
        "phase": "8B",
        "definitions": decomp_metrics["definitions"],
        "dense": phase7b_metrics["dense"],
        "dense_cross_encoder": phase7b_metrics["dense_cross_encoder"],
        "hybrid_rrf": phase7b_metrics["hybrid_rrf"],
        "hybrid_rrf_cross_encoder": phase7b_metrics["hybrid_rrf_cross_encoder"],
        "multi_query_hybrid": phase8a_metrics["multi_query_hybrid"],
        "decomposition_hybrid": decomp_metrics,
        "comparison_table": _comparison_table(phase7b_metrics, phase8a_metrics, decomp_metrics),
        "dev_metrics": decomp_metrics["by_split"].get("dev", {}),
        "challenge_metrics": decomp_metrics["by_split"].get("challenge", {}),
        "candidate_recall": {
            "candidate_union_gold_recall": _candidate_union_gold_recall(decomp_records, questions),
            "final_recall_at20": _gold_recall_at_k(decomp_records, questions, 20),
            "final_recall_at50": _gold_recall_at_k(decomp_records, questions, 50),
        },
        "hard_negative_analysis": decomp_metrics["hard_negative_analysis"],
        "decomposition_diagnostics": diagnostics,
        "inspection_cases": _inspection_report(questions, decomp_records, INSPECTION_QUESTION_IDS),
        "public_sanity": public_report,
    }

    results_path = output_root / "decomposition_results.jsonl"
    metrics_path = output_root / "metrics.json"
    packets_out_path = output_root / "generation_packets.jsonl"
    public_results_path = output_root / "public_sanity_decomposition_results.jsonl"
    public_report_path = output_root / "public_sanity_report.json"
    _write_jsonl(results_path, decomp_records)
    _write_json(metrics_path, metrics_payload)
    _write_jsonl(packets_out_path, generation_packets)
    _write_jsonl(public_results_path, public_decomp_records)
    _write_json(public_report_path, public_report)

    manifest_path = repo_root / "data/processed/manifests/phase_8b_manifest.json"
    output_paths = [
        *setup_outputs,
        repo_root / ENCODING_NORMALIZATION_REPORT_PATH,
        results_path,
        metrics_path,
        packets_out_path,
        public_results_path,
        public_report_path,
    ]
    manifest = {
        "phase": "8B",
        "version": "1.0.0",
        "status": "complete",
        "objective": "Controlled query decomposition Hybrid RRF retrieval without cross-encoder reranking",
        "configuration": {
            "per_query_dense_top_k": DENSE_TOP_K,
            "per_query_bm25_top_k": BM25_TOP_K,
            "per_query_hybrid_candidate_k": HYBRID_CANDIDATE_K,
            "second_stage_rrf_k": DECOMPOSITION_RRF_K,
            "cross_encoder_used": False,
        },
        "decomposer_prompt": {
            "version": DECOMPOSER_PROMPT_VERSION,
            "path": DECOMPOSER_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / DECOMPOSER_PROMPT_PATH),
        },
        "input_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path)
            for path in [
                questions_path,
                units_path,
                phase7b_metrics_path,
                phase8a_metrics_path,
                repo_root / QUERY_DECOMPOSITIONS_PATH,
                repo_root / PUBLIC_DECOMPOSITIONS_PATH,
                repo_root / RAW_QUERY_DECOMPOSITIONS_PATH,
                repo_root / RAW_PUBLIC_DECOMPOSITIONS_PATH,
                generator_prompt_path,
            ]
            if path.exists()
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in output_paths
        },
        "metrics": metrics_payload["comparison_table"],
        "dev_metrics": metrics_payload["dev_metrics"],
        "challenge_metrics": metrics_payload["challenge_metrics"],
        "candidate_recall": metrics_payload["candidate_recall"],
        "decomposition_diagnostics": diagnostics,
        "inspection_cases": metrics_payload["inspection_cases"],
        "public_sanity": public_report,
        "encoding_normalization": normalization_report,
        "leakage_validation": validate_packets(generation_packets, units_by_id),
    }
    _write_json(manifest_path, manifest)

    return Phase8BResult(
        status="complete",
        packets=synthetic_packets,
        public_packets=public_packets,
        outputs=output_paths,
        metrics=metrics_payload,
        manifest=manifest,
        manifest_path=manifest_path,
    )


__all__ = [
    "DECOMPOSITION_RRF_K",
    "DecompositionFilesMissingError",
    "DecompositionImportError",
    "build_decomposition_packets",
    "decomposition_rrf_score",
    "normalize_phase8b_decomposition_inputs",
    "run_phase8b",
    "validate_decompositions",
    "validate_safe_decomposition_packets",
]
