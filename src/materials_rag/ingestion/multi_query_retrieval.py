"""Phase 8A controlled multi-query retrieval infrastructure and evaluation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from materials_rag.ingestion.dense_retrieval import (
    MODEL_NAME as BI_ENCODER_MODEL_NAME,
)
from materials_rag.ingestion.dense_retrieval import (
    QUERY_PREFIX,
    _encode_texts,
    compute_metrics,
)
from materials_rag.ingestion.generation_packets import (
    PROMPT_PATH as GENERATOR_PROMPT_PATH,
)
from materials_rag.ingestion.generation_packets import (
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
from materials_rag.ingestion.hybrid_retrieval import (
    BM25_TOP_K,
    DENSE_TOP_K,
    RRF_K,
    bm25_search,
    build_bm25_index,
    fuse_dense_bm25,
)
from materials_rag.ingestion.public_sanity import PUBLIC_SANITY_QUESTIONS
from materials_rag.ingestion.qdrant_retrieval import COLLECTION_NAME, open_qdrant, qdrant_search
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

REWRITER_PROMPT_VERSION = "multi_query_rewriter_v1"
REWRITER_PROMPT_PATH = Path("docs/generation/multi_query_rewriter_v1.md")
PHASE8A_ROOT = Path("data/processed/phase8a")
QUERY_REWRITE_PACKETS_PATH = PHASE8A_ROOT / "query_rewrite_packets.jsonl"
PUBLIC_REWRITE_PACKETS_PATH = PHASE8A_ROOT / "public_sanity_query_rewrite_packets.jsonl"
QUERY_REWRITES_PATH = PHASE8A_ROOT / "query_rewrites.jsonl"
PUBLIC_REWRITES_PATH = PHASE8A_ROOT / "public_sanity_query_rewrites.jsonl"
RAW_QUERY_REWRITES_PATH = PHASE8A_ROOT / "raw/query_rewrites_raw.jsonl"
RAW_PUBLIC_REWRITES_PATH = PHASE8A_ROOT / "raw/public_sanity_query_rewrites_raw.jsonl"
ENCODING_NORMALIZATION_REPORT_PATH = PHASE8A_ROOT / "encoding_normalization_report.json"
BATCH_PROMPT_PATH = Path("experiments/17_multi_query_rewriter_batch_prompt.txt")
HYBRID_CANDIDATE_K = 50
MULTI_QUERY_RRF_K = 60
VARIANTS = ("original", "rewrite_1", "rewrite_2", "rewrite_3")
CRITICAL_FAILURE_QUESTION_IDS = ["SYNQ-002-A", "SYNQ-009-B", "SYNQ-014-B"]
KNOWN_CASE_EVIDENCE_IDS = {
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
}
MOJIBAKE_MARKERS = {
    "\u00c2",
    "\u00c3",
    "\u00e2",
    "\u20ac",
    "\u2122",
    "\u0153",
    "\ufffd",
}


class RewriteImportError(ValueError):
    """Raised when external isolated rewrite output fails validation."""


class RewriteFilesMissingError(FileNotFoundError):
    """Raised when Phase 8A rewrites have not yet been provided externally."""


@dataclass
class Phase8AResult:
    status: str
    packets: list[dict[str, Any]]
    public_packets: list[dict[str, Any]]
    outputs: list[Path]
    metrics: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    missing_rewrite_files: list[Path] | None = None


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


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def repair_text_encoding(text: str, expected_original: str | None = None) -> tuple[str, str | None]:
    """Repair deterministic UTF-8-as-cp1252 mojibake without changing semantic content."""

    try:
        candidate = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, None
    if candidate == text:
        return text, None
    if expected_original is not None and candidate == expected_original:
        return candidate, "cp1252_bytes_to_utf8_exact_original_match"
    if _mojibake_score(text) > 0 and _mojibake_score(candidate) < _mojibake_score(text):
        return candidate, "cp1252_bytes_to_utf8_reduced_mojibake"
    return text, None


def _preserve_raw_file(source: Path, raw_path: Path) -> None:
    if not source.exists():
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        shutil.copy2(source, raw_path)


def _normalize_rewrite_records(
    raw_path: Path,
    output_path: Path,
    expected_packets: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _read_jsonl(raw_path)
    expected_by_id = {packet["question_id"]: packet["original_query"] for packet in expected_packets}
    normalized_records: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for record in records:
        question_id = record.get("question_id")
        expected_original = expected_by_id.get(str(question_id))
        new_record = dict(record)
        if isinstance(new_record.get("original_query"), str):
            before = new_record["original_query"]
            after, method = repair_text_encoding(before, expected_original)
            if method:
                new_record["original_query"] = after
                changes.append(
                    {
                        "question_id": question_id,
                        "field": "original_query",
                        "before": before,
                        "after": after,
                        "repair_method": method,
                    }
                )
        if isinstance(new_record.get("rewrites"), list):
            repaired_rewrites = []
            for index, rewrite in enumerate(new_record["rewrites"]):
                if isinstance(rewrite, str):
                    after, method = repair_text_encoding(rewrite)
                    if method:
                        changes.append(
                            {
                                "question_id": question_id,
                                "field": f"rewrites[{index}]",
                                "before": rewrite,
                                "after": after,
                                "repair_method": method,
                            }
                        )
                    repaired_rewrites.append(after)
                else:
                    repaired_rewrites.append(rewrite)
            new_record["rewrites"] = repaired_rewrites
        normalized_records.append(new_record)
    _write_jsonl(output_path, normalized_records)
    return normalized_records, changes


def normalize_phase8a_rewrite_inputs(
    repo_root: Path,
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> dict[str, Any]:
    """Preserve raw rewriter output and write deterministic encoding-normalized imports."""

    query_rewrites_path = repo_root / QUERY_REWRITES_PATH
    public_rewrites_path = repo_root / PUBLIC_REWRITES_PATH
    raw_query_path = repo_root / RAW_QUERY_REWRITES_PATH
    raw_public_path = repo_root / RAW_PUBLIC_REWRITES_PATH
    _preserve_raw_file(query_rewrites_path, raw_query_path)
    _preserve_raw_file(public_rewrites_path, raw_public_path)
    changes: list[dict[str, Any]] = []
    if raw_query_path.exists():
        _, synthetic_changes = _normalize_rewrite_records(
            raw_query_path, query_rewrites_path, synthetic_packets
        )
        changes.extend(synthetic_changes)
    if raw_public_path.exists():
        _, public_changes = _normalize_rewrite_records(
            raw_public_path, public_rewrites_path, public_packets
        )
        changes.extend(public_changes)
    raw_paths = [path for path in [raw_query_path, raw_public_path] if path.exists()]
    normalized_paths = [
        path for path in [query_rewrites_path, public_rewrites_path] if path.exists()
    ]
    report = {
        "phase": "8A",
        "normalization": "deterministic transport encoding repair only",
        "raw_file_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in raw_paths
        },
        "normalized_file_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in normalized_paths
        },
        "change_count": len(changes),
        "changes": changes,
    }
    _write_json(repo_root / ENCODING_NORMALIZATION_REPORT_PATH, report)
    return report


def _safe_packet(question_id: str, query: str) -> dict[str, str]:
    return {"question_id": question_id, "original_query": query}


def build_rewrite_packets(repo_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    questions = _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl")
    synthetic_packets = [_safe_packet(row["question_id"], row["query"]) for row in questions]
    public_packets = [
        _safe_packet(row["question_id"], row["query"]) for row in PUBLIC_SANITY_QUESTIONS
    ]
    validate_safe_packets(synthetic_packets)
    validate_safe_packets(public_packets)
    return synthetic_packets, public_packets


def validate_safe_packets(packets: list[dict[str, Any]]) -> None:
    for packet in packets:
        if set(packet) != {"question_id", "original_query"}:
            raise ValueError(f"Unsafe rewrite packet keys: {sorted(packet)}")
        if not packet["question_id"] or not packet["original_query"]:
            raise ValueError("Rewrite packets require non-empty question_id and original_query.")
        if LEAKAGE_PACKET_KEYS & set(packet):
            raise ValueError(f"Rewrite packet contains leakage keys: {LEAKAGE_PACKET_KEYS & set(packet)}")


def write_rewriter_batch_prompt(
    repo_root: Path,
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> Path:
    prompt_text = (repo_root / REWRITER_PROMPT_PATH).read_text(encoding="utf-8").strip()
    lines = [
        prompt_text,
        "",
        "Rewrite the following query-only requests. Return one JSON object per input object, in the same order.",
        "Do not add explanations or markdown.",
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


def validate_rewrites(
    rewrite_path: Path,
    expected_packets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not rewrite_path.exists():
        raise RewriteFilesMissingError(f"Missing external rewrite file: {rewrite_path}")
    records = _read_jsonl(rewrite_path)
    expected_ids = [packet["question_id"] for packet in expected_packets]
    actual_ids = [record.get("question_id") for record in records]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise RewriteImportError(
            "Rewrite IDs must match expected packet order exactly. "
            f"missing={missing}; unexpected={unexpected}."
        )
    expected_by_id = {packet["question_id"]: packet for packet in expected_packets}
    validated: list[dict[str, Any]] = []
    for record in records:
        if set(record) != {"question_id", "original_query", "rewrites"}:
            raise RewriteImportError(
                f"{record.get('question_id')} has invalid keys: {sorted(record)}"
            )
        question_id = record["question_id"]
        expected_query = expected_by_id[question_id]["original_query"]
        if record["original_query"] != expected_query:
            raise RewriteImportError(f"{question_id} original_query does not match frozen text.")
        rewrites = record["rewrites"]
        if not isinstance(rewrites, list) or len(rewrites) != 3:
            raise RewriteImportError(f"{question_id} must contain exactly three rewrites.")
        normalized = []
        for rewrite in rewrites:
            if not isinstance(rewrite, str) or not rewrite.strip():
                raise RewriteImportError(f"{question_id} contains an empty or non-string rewrite.")
            text = rewrite.strip()
            if text == expected_query:
                raise RewriteImportError(f"{question_id} rewrite is identical to the original query.")
            normalized.append(text)
        if len(set(normalized)) != 3:
            raise RewriteImportError(f"{question_id} rewrites must be unique.")
        validated.append(
            {"question_id": question_id, "original_query": expected_query, "rewrites": normalized}
        )
    return validated


def _variant_queries(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"variant": "original", "query": record["original_query"]},
        {"variant": "rewrite_1", "query": record["rewrites"][0]},
        {"variant": "rewrite_2", "query": record["rewrites"][1]},
        {"variant": "rewrite_3", "query": record["rewrites"][2]},
    ]


def _dense_for_variant_queries(repo_root: Path, variant_rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
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


def _hybrid_rankings_for_rewrites(
    repo_root: Path,
    rewrite_records: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units_by_id = {unit["id"]: unit for unit in units}
    bm25_index = build_bm25_index(units)
    variant_rows = []
    for record in rewrite_records:
        for variant in _variant_queries(record):
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
    return [_fuse_multi_query(record, hybrid_by_variant) for record in rewrite_records]


def multi_query_rrf_score(variant_ranks: dict[str, int | None], rrf_k: int = MULTI_QUERY_RRF_K) -> float:
    return float(
        sum(1.0 / (rrf_k + rank) for rank in variant_ranks.values() if rank is not None)
    )


def _fuse_multi_query(
    rewrite_record: dict[str, Any],
    hybrid_by_variant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variant_results = {
        variant: hybrid_by_variant[f"{rewrite_record['question_id']}::{variant}"]["results"][:HYBRID_CANDIDATE_K]
        for variant in VARIANTS
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
        variant_ranks = {
            variant: by_variant_id[variant].get(unit_id, {}).get("hybrid_rank")
            for variant in VARIANTS
        }
        exemplar = next(by_variant_id[variant][unit_id] for variant in VARIANTS if unit_id in by_variant_id[variant])
        fused.append(
            {
                "retrieval_unit_id": unit_id,
                "multi_query_rrf_score": multi_query_rrf_score(variant_ranks),
                "original_hybrid_rank": variant_ranks["original"],
                "rewrite_1_hybrid_rank": variant_ranks["rewrite_1"],
                "rewrite_2_hybrid_rank": variant_ranks["rewrite_2"],
                "rewrite_3_hybrid_rank": variant_ranks["rewrite_3"],
                "query_variant_hits": [
                    {
                        "variant": variant,
                        "hybrid_rank": variant_ranks[variant],
                        "rrf_score": by_variant_id[variant].get(unit_id, {}).get("rrf_score"),
                    }
                    for variant in VARIANTS
                    if unit_id in by_variant_id[variant]
                ],
                "kind": exemplar["kind"],
                "source": exemplar["source"],
                "evidence_id": exemplar.get("evidence_id"),
                "document_id": exemplar.get("document_id"),
            }
        )
    fused.sort(
        key=lambda row: (
            -row["multi_query_rrf_score"],
            row["original_hybrid_rank"] if row["original_hybrid_rank"] is not None else 999,
            row["rewrite_1_hybrid_rank"] if row["rewrite_1_hybrid_rank"] is not None else 999,
            row["retrieval_unit_id"],
        )
    )
    for rank, row in enumerate(fused, start=1):
        row["multi_query_rank"] = rank
    return {
        "question_id": rewrite_record["question_id"],
        "query": rewrite_record["original_query"],
        "rewrite_count": 3,
        "hybrid_candidate_k_per_query": HYBRID_CANDIDATE_K,
        "results": fused,
    }


def _ranked_for_metrics(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["question_id"]: [
            {
                "retrieval_unit_id": result["retrieval_unit_id"],
                "score": result["multi_query_rrf_score"],
            }
            for result in sorted(row["results"], key=lambda result: result["multi_query_rank"])
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
            row["retrieval_unit_id"]
            for row in by_question[question["question_id"]]["results"][:k]
        ]
        hits = [unit_id for unit_id in gold_ids if unit_id in set(ranked_ids)]
        missing = [unit_id for unit_id in gold_ids if unit_id not in set(ranked_ids)]
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


def _comparison_table(phase7b_metrics: dict[str, Any], mq_metrics: dict[str, Any]) -> list[dict[str, float]]:
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
    return [
        {
            "metric": metric,
            "dense": phase7b_metrics["dense"]["overall_non_abstain"][metric],
            "dense_cross_encoder": phase7b_metrics["dense_cross_encoder"]["overall_non_abstain"][metric],
            "hybrid_rrf": phase7b_metrics["hybrid_rrf"]["overall_non_abstain"][metric],
            "hybrid_rrf_cross_encoder": phase7b_metrics["hybrid_rrf_cross_encoder"][
                "overall_non_abstain"
            ][metric],
            "multi_query_hybrid": mq_metrics["overall_non_abstain"][metric],
        }
        for metric in ordered
    ]


def _rank_for_evidence(row: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    for result in row["results"]:
        if result.get("evidence_id") == evidence_id:
            return {
                "retrieval_unit_id": result["retrieval_unit_id"],
                "multi_query_rank": result["multi_query_rank"],
                "multi_query_rrf_score": result["multi_query_rrf_score"],
                "original_hybrid_rank": result["original_hybrid_rank"],
                "rewrite_1_hybrid_rank": result["rewrite_1_hybrid_rank"],
                "rewrite_2_hybrid_rank": result["rewrite_2_hybrid_rank"],
                "rewrite_3_hybrid_rank": result["rewrite_3_hybrid_rank"],
            }
    return None


def _critical_failure_report(
    questions: list[dict[str, Any]],
    phase7a_records: list[dict[str, Any]],
    mq_records: list[dict[str, Any]],
) -> dict[str, Any]:
    questions_by_id = {row["question_id"]: row for row in questions}
    phase7a_by_id = {row["question_id"]: row for row in phase7a_records}
    mq_by_id = {row["question_id"]: row for row in mq_records}
    report = {}
    for question_id in CRITICAL_FAILURE_QUESTION_IDS:
        gold_ids = questions_by_id[question_id].get("gold_chunk_ids", [])
        original_by_unit = {
            result["retrieval_unit_id"]: result for result in phase7a_by_id[question_id]["results"]
        }
        mq_by_unit = {result["retrieval_unit_id"]: result for result in mq_by_id[question_id]["results"]}
        report[question_id] = [
            {
                "retrieval_unit_id": unit_id,
                "missing_from_original_hybrid_top50": original_by_unit.get(unit_id, {}).get(
                    "hybrid_rank"
                )
                is None
                or original_by_unit.get(unit_id, {}).get("hybrid_rank", 999) > HYBRID_CANDIDATE_K,
                "original_hybrid_rank": original_by_unit.get(unit_id, {}).get("hybrid_rank"),
                "rewrite_1_hybrid_rank": mq_by_unit.get(unit_id, {}).get("rewrite_1_hybrid_rank"),
                "rewrite_2_hybrid_rank": mq_by_unit.get(unit_id, {}).get("rewrite_2_hybrid_rank"),
                "rewrite_3_hybrid_rank": mq_by_unit.get(unit_id, {}).get("rewrite_3_hybrid_rank"),
                "multi_query_rank": mq_by_unit.get(unit_id, {}).get("multi_query_rank"),
            }
            for unit_id in gold_ids
        ]
    return report


def _known_case_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    return {
        question_id: {
            evidence_id: _rank_for_evidence(by_question[question_id], evidence_id)
            for evidence_id in evidence_ids
        }
        for question_id, evidence_ids in KNOWN_CASE_EVIDENCE_IDS.items()
    }


def _packet_for_generation(
    row: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> dict[str, Any]:
    ordered = sorted(row["results"], key=lambda result: result["multi_query_rank"])[:GENERATION_TOP_K]
    retrieval_unit_ids = [result["retrieval_unit_id"] for result in ordered]
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in retrieval_unit_ids]
    generator_input = format_generator_input(row["query"], context_items)
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "top_k": GENERATION_TOP_K,
        "retrieval_unit_ids": retrieval_unit_ids,
        "retrieval_backend": "Phase 8A Multi-query Hybrid RRF",
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
    packets = [_packet_for_generation(row, units_by_id, prompt_hash) for row in records]
    validation = validate_packets(packets, units_by_id)
    if not validation["passed"]:
        raise ValueError(f"Phase 8A generation packet validation failed: {validation}")
    for packet in packets:
        lowered = packet["generator_input"].lower()
        for term in ["rewrite_", "multi_query", "rrf_score", "hybrid_rank", "score:"]:
            if term in lowered:
                raise ValueError(f"Phase 8A packet exposes retrieval internals: {term}")
    return packets


def _public_sanity_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    rows = []
    for question in PUBLIC_SANITY_QUESTIONS:
        result_row = by_question[question["question_id"]]
        expected = set(question["expected_retrieval_unit_ids"])
        ranked = sorted(result_row["results"], key=lambda row: row["multi_query_rank"])
        expected_rows = [row for row in ranked if row["retrieval_unit_id"] in expected]
        best = expected_rows[0] if expected_rows else None
        rows.append(
            {
                "question_id": question["question_id"],
                "query": question["query"],
                "expected_retrieval_unit_ids": question["expected_retrieval_unit_ids"],
                "best_expected_multi_query_rank": best.get("multi_query_rank") if best else None,
                "best_expected_variant_ranks": {
                    key: best.get(key) if best else None
                    for key in [
                        "original_hybrid_rank",
                        "rewrite_1_hybrid_rank",
                        "rewrite_2_hybrid_rank",
                        "rewrite_3_hybrid_rank",
                    ]
                },
                "multi_query_top5_ids": [row["retrieval_unit_id"] for row in ranked[:5]],
            }
        )
    return {
        "question_count": len(rows),
        "pubsan_005": next(row for row in rows if row["question_id"] == "PUBSAN-005"),
        "per_question": rows,
    }


def run_phase8a(repo_root: Path | None = None) -> Phase8AResult:
    """Prepare safe rewrite inputs and run Phase 8A when external rewrites exist."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / PHASE8A_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    synthetic_packets, public_packets = build_rewrite_packets(repo_root)
    packets_path = repo_root / QUERY_REWRITE_PACKETS_PATH
    public_packets_path = repo_root / PUBLIC_REWRITE_PACKETS_PATH
    _write_jsonl(packets_path, synthetic_packets)
    _write_jsonl(public_packets_path, public_packets)
    batch_prompt_path = write_rewriter_batch_prompt(repo_root, synthetic_packets, public_packets)

    rewrite_paths = [repo_root / QUERY_REWRITES_PATH, repo_root / PUBLIC_REWRITES_PATH]
    missing = [path for path in rewrite_paths if not path.exists()]
    setup_outputs = [packets_path, public_packets_path, batch_prompt_path]
    if missing:
        return Phase8AResult(
            status="waiting_for_external_rewrites",
            packets=synthetic_packets,
            public_packets=public_packets,
            outputs=setup_outputs,
            missing_rewrite_files=missing,
        )

    normalization_report = normalize_phase8a_rewrite_inputs(
        repo_root, synthetic_packets, public_packets
    )
    rewrites = validate_rewrites(repo_root / QUERY_REWRITES_PATH, synthetic_packets)
    public_rewrites = validate_rewrites(repo_root / PUBLIC_REWRITES_PATH, public_packets)
    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    phase7a_results_path = repo_root / "data/processed/phase7a/hybrid_results.jsonl"
    phase7b_metrics_path = repo_root / "data/processed/phase7b/metrics.json"
    generator_prompt_path = repo_root / GENERATOR_PROMPT_PATH
    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    units_by_id = {unit["id"]: unit for unit in units}
    phase7a_records = _read_jsonl(phase7a_results_path)
    phase7b_metrics = _read_json(phase7b_metrics_path)

    mq_records = _hybrid_rankings_for_rewrites(repo_root, rewrites, units)
    public_mq_records = _hybrid_rankings_for_rewrites(repo_root, public_rewrites, units)
    mq_metrics = compute_metrics(questions, _ranked_for_metrics(mq_records))
    generator_prompt_hash = _sha256_text(generator_prompt_path.read_text(encoding="utf-8"))
    generation_packets = _generation_packets(mq_records, units_by_id, generator_prompt_hash)
    public_report = _public_sanity_report(public_mq_records)

    metrics_payload = {
        "phase": "8A",
        "definitions": mq_metrics["definitions"],
        "dense": phase7b_metrics["dense"],
        "dense_cross_encoder": phase7b_metrics["dense_cross_encoder"],
        "hybrid_rrf": phase7b_metrics["hybrid_rrf"],
        "hybrid_rrf_cross_encoder": phase7b_metrics["hybrid_rrf_cross_encoder"],
        "multi_query_hybrid": mq_metrics,
        "comparison_table": _comparison_table(phase7b_metrics, mq_metrics),
        "dev_metrics": mq_metrics["by_split"].get("dev", {}),
        "challenge_metrics": mq_metrics["by_split"].get("challenge", {}),
        "candidate_recall": {
            "recall_at20": _gold_recall_at_k(mq_records, questions, 20),
            "recall_at50": _gold_recall_at_k(mq_records, questions, 50),
        },
        "hard_negative_analysis": mq_metrics["hard_negative_analysis"],
        "critical_failure_cases": _critical_failure_report(questions, phase7a_records, mq_records),
        "known_cases": _known_case_report(mq_records),
        "public_sanity": public_report,
    }

    results_path = output_root / "multi_query_results.jsonl"
    metrics_path = output_root / "metrics.json"
    packets_out_path = output_root / "generation_packets.jsonl"
    public_results_path = output_root / "public_sanity_multi_query_results.jsonl"
    public_report_path = output_root / "public_sanity_report.json"
    _write_jsonl(results_path, mq_records)
    _write_json(metrics_path, metrics_payload)
    _write_jsonl(packets_out_path, generation_packets)
    _write_jsonl(public_results_path, public_mq_records)
    _write_json(public_report_path, public_report)

    manifest_path = repo_root / "data/processed/manifests/phase_8a_manifest.json"
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
        "phase": "8A",
        "version": "1.0.0",
        "objective": "Controlled multi-query Hybrid RRF retrieval without cross-encoder reranking",
        "configuration": {
            "rewrite_count": 3,
            "query_variants": list(VARIANTS),
            "per_query_dense_top_k": DENSE_TOP_K,
            "per_query_bm25_top_k": BM25_TOP_K,
            "per_query_hybrid_candidate_k": HYBRID_CANDIDATE_K,
            "second_stage_rrf_k": MULTI_QUERY_RRF_K,
            "cross_encoder_used": False,
        },
        "rewriter_prompt": {
            "version": REWRITER_PROMPT_VERSION,
            "path": REWRITER_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / REWRITER_PROMPT_PATH),
        },
        "input_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path)
            for path in [
                questions_path,
                units_path,
                phase7a_results_path,
                phase7b_metrics_path,
                repo_root / QUERY_REWRITES_PATH,
                repo_root / PUBLIC_REWRITES_PATH,
                repo_root / RAW_QUERY_REWRITES_PATH,
                repo_root / RAW_PUBLIC_REWRITES_PATH,
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
        "critical_failure_cases": metrics_payload["critical_failure_cases"],
        "known_cases": metrics_payload["known_cases"],
        "public_sanity": public_report,
        "encoding_normalization": normalization_report,
        "leakage_validation": validate_packets(generation_packets, units_by_id),
    }
    _write_json(manifest_path, manifest)

    return Phase8AResult(
        status="complete",
        packets=synthetic_packets,
        public_packets=public_packets,
        outputs=output_paths,
        metrics=metrics_payload,
        manifest=manifest,
        manifest_path=manifest_path,
    )


__all__ = [
    "MULTI_QUERY_RRF_K",
    "RewriteFilesMissingError",
    "RewriteImportError",
    "build_rewrite_packets",
    "multi_query_rrf_score",
    "normalize_phase8a_rewrite_inputs",
    "repair_text_encoding",
    "run_phase8a",
    "validate_rewrites",
    "validate_safe_packets",
]

