"""Phase 7A hybrid dense + BM25 retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
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
from materials_rag.ingestion.qdrant_retrieval import (
    COLLECTION_NAME,
    load_phase5b_inputs,
    open_qdrant,
    qdrant_search,
)
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

DENSE_TOP_K = 50
BM25_TOP_K = 50
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_PATTERN = re.compile(r"(?u)\b[\w]+(?:[-./][\w]+)*\b")
KNOWN_CASE_EVIDENCE_IDS = {
    "SYNQ-001-A": ["FAT-B017#results", "FAT-B017#conditions"],
}


@dataclass
class BM25Index:
    units: list[dict[str, Any]]
    tokenized_documents: list[list[str]]
    term_frequencies: list[Counter[str]]
    document_frequencies: dict[str, int]
    document_lengths: list[int]
    average_document_length: float
    idf: dict[str, float]


@dataclass
class Phase7AResult:
    bm25_results: list[dict[str, Any]]
    hybrid_results: list[dict[str, Any]]
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


def tokenize(text: str) -> list[str]:
    """Deterministic Unicode-aware tokenizer that preserves useful identifiers."""

    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0).strip("_")
        if not token:
            continue
        tokens.append(token)
        if any(separator in token for separator in "-./"):
            tokens.extend(part for part in re.split(r"[-./]+", token) if part)
    for first, second in pairwise(list(tokens)):
        if first.isalpha() and second.isdigit():
            tokens.append(f"{first}_{second}")
    return tokens


def build_bm25_index(units: list[dict[str, Any]]) -> BM25Index:
    tokenized_documents = [tokenize(str(unit["retrieval_text"])) for unit in units]
    term_frequencies = [Counter(tokens) for tokens in tokenized_documents]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    if not document_lengths or any(length == 0 for length in document_lengths):
        raise ValueError("BM25 documents must be non-empty after tokenization.")
    document_frequencies: dict[str, int] = {}
    for frequencies in term_frequencies:
        for token in frequencies:
            document_frequencies[token] = document_frequencies.get(token, 0) + 1
    n_documents = len(units)
    idf = {
        token: math.log(1.0 + (n_documents - df + 0.5) / (df + 0.5))
        for token, df in document_frequencies.items()
    }
    return BM25Index(
        units=units,
        tokenized_documents=tokenized_documents,
        term_frequencies=term_frequencies,
        document_frequencies=document_frequencies,
        document_lengths=document_lengths,
        average_document_length=float(sum(document_lengths) / len(document_lengths)),
        idf=idf,
    )


def _bm25_score(query_tokens: list[str], doc_index: int, index: BM25Index) -> float:
    frequencies = index.term_frequencies[doc_index]
    doc_length = index.document_lengths[doc_index]
    score = 0.0
    for token in query_tokens:
        frequency = frequencies.get(token, 0)
        if frequency == 0:
            continue
        denominator = frequency + BM25_K1 * (
            1.0 - BM25_B + BM25_B * doc_length / index.average_document_length
        )
        score += index.idf.get(token, 0.0) * (frequency * (BM25_K1 + 1.0)) / denominator
    return float(score)


def _result_metadata(unit: dict[str, Any]) -> dict[str, Any]:
    metadata = unit.get("metadata", {})
    return {
        "kind": unit["kind"],
        "source": unit["source"],
        "evidence_id": metadata.get("evidence_id"),
        "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
    }


def bm25_search(index: BM25Index, query: str, top_k: int = BM25_TOP_K) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    query_counts = Counter(query_tokens)
    scored = []
    for doc_index, unit in enumerate(index.units):
        score = _bm25_score(list(query_counts.elements()), doc_index, index)
        scored.append((score, doc_index, unit))
    scored.sort(key=lambda item: (-item[0], item[1]))
    results: list[dict[str, Any]] = []
    for rank, (score, _doc_index, unit) in enumerate(scored[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "retrieval_unit_id": unit["id"],
                "score": float(score),
                **_result_metadata(unit),
            }
        )
    return results


def _dense_top50_records(repo_root: Path, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs = load_phase5b_inputs(repo_root)
    if [question["question_id"] for question in questions] != [
        question["question_id"] for question in inputs["questions"]
    ]:
        raise ValueError("Phase 4 questions and Phase 5A query embeddings are not aligned.")
    client = open_qdrant(repo_root / "data/processed/phase5b/qdrant_storage")
    try:
        return [
            {
                "question_id": question["question_id"],
                "query_embedding_row_index": query_index,
                "results": qdrant_search(
                    client,
                    COLLECTION_NAME,
                    inputs["query_matrix"][query_index],
                    DENSE_TOP_K,
                ),
            }
            for query_index, question in enumerate(questions)
        ]
    finally:
        client.close()


def _bm25_records(
    questions: list[dict[str, Any]],
    index: BM25Index,
    top_k: int = BM25_TOP_K,
) -> list[dict[str, Any]]:
    return [
        {
            "question_id": question["question_id"],
            "query": question["query"],
            "results": bm25_search(index, question["query"], top_k),
        }
        for question in questions
    ]


def rrf_score(
    dense_rank: int | None,
    bm25_rank: int | None,
    rrf_k: int = RRF_K,
) -> float:
    score = 0.0
    if dense_rank is not None:
        score += 1.0 / (rrf_k + dense_rank)
    if bm25_rank is not None:
        score += 1.0 / (rrf_k + bm25_rank)
    return float(score)


def fuse_dense_bm25(
    question: dict[str, Any],
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    rrf_k: int = RRF_K,
) -> dict[str, Any]:
    dense_by_id = {row["retrieval_unit_id"]: row for row in dense_results}
    bm25_by_id = {row["retrieval_unit_id"]: row for row in bm25_results}
    candidate_ids = sorted(set(dense_by_id) | set(bm25_by_id))
    fused_rows: list[dict[str, Any]] = []
    for unit_id in candidate_ids:
        dense_row = dense_by_id.get(unit_id)
        bm25_row = bm25_by_id.get(unit_id)
        unit = units_by_id[unit_id]
        dense_rank = int(dense_row["rank"]) if dense_row else None
        bm25_rank = int(bm25_row["rank"]) if bm25_row else None
        fused_rows.append(
            {
                "retrieval_unit_id": unit_id,
                "rrf_score": rrf_score(dense_rank, bm25_rank, rrf_k),
                "dense_rank": dense_rank,
                "dense_score": float(dense_row["score"]) if dense_row else None,
                "bm25_rank": bm25_rank,
                "bm25_score": float(bm25_row["score"]) if bm25_row else None,
                **_result_metadata(unit),
            }
        )
    fused_rows.sort(
        key=lambda row: (
            -row["rrf_score"],
            row["dense_rank"] if row["dense_rank"] is not None else DENSE_TOP_K + 1,
            row["bm25_rank"] if row["bm25_rank"] is not None else BM25_TOP_K + 1,
            row["retrieval_unit_id"],
        )
    )
    for rank, row in enumerate(fused_rows, start=1):
        row["hybrid_rank"] = rank
    return {
        "question_id": question["question_id"],
        "query": question["query"],
        "rrf_k": rrf_k,
        "dense_top_k": len(dense_results),
        "bm25_top_k": len(bm25_results),
        "candidate_union_size": len(fused_rows),
        "results": fused_rows,
    }


def _hybrid_records(
    questions: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    bm25_records_: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    dense_by_question = {row["question_id"]: row for row in dense_records}
    bm25_by_question = {row["question_id"]: row for row in bm25_records_}
    return [
        fuse_dense_bm25(
            question,
            dense_by_question[question["question_id"]]["results"],
            bm25_by_question[question["question_id"]]["results"],
            units_by_id,
            rrf_k,
        )
        for question in questions
    ]


def _ranked_for_metrics(
    records: list[dict[str, Any]],
    rank_key: str,
    score_key: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        row["question_id"]: [
            {"retrieval_unit_id": result["retrieval_unit_id"], "score": result[score_key]}
            for result in sorted(row["results"], key=lambda item: item[rank_key])
        ]
        for row in records
    }


def _first_gold_rank(ranked_ids: list[str], gold_ids: list[str]) -> int | None:
    gold = set(gold_ids)
    return next((index + 1 for index, unit_id in enumerate(ranked_ids) if unit_id in gold), None)


def _gold_recall_at_k(records: list[dict[str, Any]], questions: list[dict[str, Any]], k: int) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    per_question: list[dict[str, Any]] = []
    recalls: list[float] = []
    for question in questions:
        gold_ids = list(question.get("gold_chunk_ids", []))
        ranked_ids = [row["retrieval_unit_id"] for row in by_question[question["question_id"]]["results"][:k]]
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


def _candidate_recall_report(
    questions: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    bm25_records_: list[dict[str, Any]],
    hybrid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_by_question = {row["question_id"]: row for row in dense_records}
    bm25_by_question = {row["question_id"]: row for row in bm25_records_}
    bm25_finds_dense_misses: list[dict[str, Any]] = []
    for question in questions:
        gold_ids = set(question.get("gold_chunk_ids", []))
        dense20 = {row["retrieval_unit_id"] for row in dense_by_question[question["question_id"]]["results"][:20]}
        dense50 = {row["retrieval_unit_id"] for row in dense_by_question[question["question_id"]]["results"][:50]}
        bm25_ids = {row["retrieval_unit_id"] for row in bm25_by_question[question["question_id"]]["results"][:50]}
        found_vs_top20 = sorted((gold_ids - dense20) & bm25_ids)
        found_vs_top50 = sorted((gold_ids - dense50) & bm25_ids)
        if found_vs_top20 or found_vs_top50:
            bm25_finds_dense_misses.append(
                {
                    "question_id": question["question_id"],
                    "bm25_gold_found_that_dense_top20_missed": found_vs_top20,
                    "bm25_gold_found_that_dense_top50_missed": found_vs_top50,
                }
            )
    return {
        "dense": {
            "recall_at20": _gold_recall_at_k(dense_records, questions, 20),
            "recall_at50": _gold_recall_at_k(dense_records, questions, 50),
        },
        "bm25": {
            "recall_at20": _gold_recall_at_k(bm25_records_, questions, 20),
            "recall_at50": _gold_recall_at_k(bm25_records_, questions, 50),
        },
        "hybrid": {
            "recall_at20": _gold_recall_at_k(hybrid_records, questions, 20),
            "recall_at50": _gold_recall_at_k(hybrid_records, questions, 50),
        },
        "bm25_finds_gold_dense_misses": bm25_finds_dense_misses,
    }


def _comparison_table(
    dense_metrics: dict[str, Any],
    bm25_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
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
    bm25 = bm25_metrics["overall_non_abstain"]
    hybrid = hybrid_metrics["overall_non_abstain"]
    return [
        {
            "metric": metric,
            "dense_baseline": float(dense[metric]),
            "bm25_only": float(bm25[metric]),
            "hybrid_rrf": float(hybrid[metric]),
            "hybrid_delta_vs_dense": float(hybrid[metric] - dense[metric]),
            "hybrid_delta_vs_bm25": float(hybrid[metric] - bm25[metric]),
        }
        for metric in ordered
    ]


def _rank_for_evidence(row: dict[str, Any], evidence_id: str, rank_key: str) -> dict[str, Any] | None:
    for result in row["results"]:
        if result.get("evidence_id") == evidence_id:
            return {
                "retrieval_unit_id": result["retrieval_unit_id"],
                "rank": result[rank_key],
                "dense_rank": result.get("dense_rank"),
                "bm25_rank": result.get("bm25_rank"),
                "hybrid_rank": result.get("hybrid_rank"),
                "rrf_score": result.get("rrf_score"),
                "dense_score": result.get("dense_score"),
                "bm25_score": result.get("bm25_score"),
            }
    return None


def _case_studies(
    questions: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    bm25_records_: list[dict[str, Any]],
    hybrid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_by_question = {row["question_id"]: row for row in dense_records}
    bm25_by_question = {row["question_id"]: row for row in bm25_records_}
    hybrid_by_question = {row["question_id"]: row for row in hybrid_records}
    b017 = {
        evidence_id: {
            "dense": _rank_for_evidence(dense_by_question["SYNQ-001-A"], evidence_id, "rank"),
            "bm25": _rank_for_evidence(bm25_by_question["SYNQ-001-A"], evidence_id, "rank"),
            "hybrid": _rank_for_evidence(hybrid_by_question["SYNQ-001-A"], evidence_id, "hybrid_rank"),
        }
        for evidence_id in KNOWN_CASE_EVIDENCE_IDS["SYNQ-001-A"]
    }

    examples: dict[str, Any] = {}
    for question in questions:
        gold_ids = list(question.get("gold_chunk_ids", []))
        if not gold_ids:
            continue
        question_id = question["question_id"]
        dense_rank = _first_gold_rank(
            [row["retrieval_unit_id"] for row in dense_by_question[question_id]["results"]], gold_ids
        )
        bm25_rank = _first_gold_rank(
            [row["retrieval_unit_id"] for row in bm25_by_question[question_id]["results"]], gold_ids
        )
        hybrid_rank = _first_gold_rank(
            [row["retrieval_unit_id"] for row in hybrid_by_question[question_id]["results"]], gold_ids
        )
        dense_value = dense_rank if dense_rank is not None else 999
        bm25_value = bm25_rank if bm25_rank is not None else 999
        hybrid_value = hybrid_rank if hybrid_rank is not None else 999
        payload = {
            "question_id": question_id,
            "query": question["query"],
            "first_gold_dense_rank": dense_rank,
            "first_gold_bm25_rank": bm25_rank,
            "first_gold_hybrid_rank": hybrid_rank,
        }
        if "dense_wins_clearly" not in examples and dense_value + 5 <= bm25_value:
            examples["dense_wins_clearly"] = payload
        if "bm25_wins_clearly" not in examples and bm25_value + 5 <= dense_value:
            examples["bm25_wins_clearly"] = payload
        if "hybrid_improves_or_preserves_stronger_branch" not in examples and hybrid_value <= min(
            dense_value, bm25_value
        ):
            examples["hybrid_improves_or_preserves_stronger_branch"] = payload
        if len(examples) == 3:
            break
    return {"b017_f03": b017, "examples": examples}


def _packet_for_hybrid_question(
    row: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> dict[str, Any]:
    ordered = sorted(row["results"], key=lambda result: result["hybrid_rank"])[:GENERATION_TOP_K]
    retrieval_unit_ids = [result["retrieval_unit_id"] for result in ordered]
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in retrieval_unit_ids]
    generator_input = format_generator_input(row["query"], context_items)
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "top_k": GENERATION_TOP_K,
        "retrieval_unit_ids": retrieval_unit_ids,
        "retrieval_backend": "Phase 7A hybrid dense + BM25 RRF",
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def _generation_packets(
    hybrid_records: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> list[dict[str, Any]]:
    packets = [_packet_for_hybrid_question(row, units_by_id, prompt_hash) for row in hybrid_records]
    validation = validate_packets(packets, units_by_id)
    if not validation["passed"]:
        raise ValueError(f"Phase 7A generation packet validation failed: {validation}")
    for packet in packets:
        lowered = packet["generator_input"].lower()
        for term in ["bm25", "rrf", "dense score", "bm25 score", "rrf score"]:
            if term in lowered:
                raise ValueError(f"Generator packet exposes retrieval internals: {term}")
    return packets


def _source_path_violations(records: list[dict[str, Any]]) -> list[str]:
    violations = []
    for row in records:
        for result in row["results"]:
            source_path = str(result.get("source", {}).get("source_path", ""))
            if any(source_path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
                violations.append(f"{row['question_id']}:{result['retrieval_unit_id']}:{source_path}")
    return violations


def _public_sanity_dense_records(repo_root: Path) -> list[dict[str, Any]]:
    model = SentenceTransformer(BI_ENCODER_MODEL_NAME, local_files_only=True)
    model.eval()
    query_matrix = _encode_texts(
        model,
        [QUERY_PREFIX + question["query"] for question in PUBLIC_SANITY_QUESTIONS],
    )
    client = open_qdrant(repo_root / "data/processed/phase5b/qdrant_storage")
    try:
        return [
            {
                "question_id": question["question_id"],
                "results": qdrant_search(client, COLLECTION_NAME, query_matrix[index], DENSE_TOP_K),
            }
            for index, question in enumerate(PUBLIC_SANITY_QUESTIONS)
        ]
    finally:
        client.close()


def _public_sanity_report(
    repo_root: Path,
    index: BM25Index,
    units_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    questions = [
        {
            "question_id": question["question_id"],
            "query": question["query"],
            "expected_retrieval_unit_ids": question["expected_retrieval_unit_ids"],
            "source_type": question["source_type"],
        }
        for question in PUBLIC_SANITY_QUESTIONS
    ]
    dense = _public_sanity_dense_records(repo_root)
    bm25 = _bm25_records(questions, index, BM25_TOP_K)
    hybrid = _hybrid_records(questions, dense, bm25, units_by_id, RRF_K)
    rows = []
    for question, dense_row, bm25_row, hybrid_row in zip(questions, dense, bm25, hybrid, strict=True):
        expected = set(question["expected_retrieval_unit_ids"])
        dense_ids = [row["retrieval_unit_id"] for row in dense_row["results"]]
        bm25_ids = [row["retrieval_unit_id"] for row in bm25_row["results"]]
        hybrid_ids = [row["retrieval_unit_id"] for row in hybrid_row["results"]]

        def best_rank(ids: list[str], expected_ids: set[str] = expected) -> int | None:
            ranks = [index + 1 for index, unit_id in enumerate(ids) if unit_id in expected_ids]
            return min(ranks) if ranks else None

        rows.append(
            {
                "question_id": question["question_id"],
                "query": question["query"],
                "source_type": question["source_type"],
                "expected_retrieval_unit_ids": sorted(expected),
                "best_expected_dense_rank": best_rank(dense_ids),
                "best_expected_bm25_rank": best_rank(bm25_ids),
                "best_expected_hybrid_rank": best_rank(hybrid_ids),
                "expected_in_dense_top5": (best_rank(dense_ids) or 999) <= 5,
                "expected_in_bm25_top5": (best_rank(bm25_ids) or 999) <= 5,
                "expected_in_hybrid_top5": (best_rank(hybrid_ids) or 999) <= 5,
                "dense_top5_ids": dense_ids[:5],
                "bm25_top5_ids": bm25_ids[:5],
                "hybrid_top5_ids": hybrid_ids[:5],
            }
        )
    pubsan_005 = next(row for row in rows if row["question_id"] == "PUBSAN-005")
    return bm25, hybrid, {"question_count": len(rows), "pubsan_005": pubsan_005, "per_question": rows}


def run_phase7a(repo_root: Path | None = None, rrf_k: int = RRF_K) -> Phase7AResult:
    """Run BM25 and dense+BM25 RRF hybrid retrieval over the Phase 4.5 corpus."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / "data/processed/phase7a"
    output_root.mkdir(parents=True, exist_ok=True)

    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    phase5b_manifest_path = repo_root / "data/processed/manifests/phase_5b_manifest.json"
    prompt_path = repo_root / PROMPT_PATH

    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    units_by_id = {unit["id"]: unit for unit in units}
    if len(units) != 277 or len(units_by_id) != 277:
        raise ValueError("Phase 7A requires exactly 277 unique Phase 4.5 retrieval units.")

    index = build_bm25_index(units)
    dense_records = _dense_top50_records(repo_root, questions)
    bm25_records_ = _bm25_records(questions, index, BM25_TOP_K)
    hybrid_records = _hybrid_records(questions, dense_records, bm25_records_, units_by_id, rrf_k)
    source_violations = _source_path_violations(bm25_records_) + _source_path_violations(hybrid_records)
    if source_violations:
        raise ValueError(f"Excluded authoring/eval content entered Phase 7A candidates: {source_violations}")

    dense_metrics = compute_metrics(questions, _ranked_for_metrics(dense_records, "rank", "score"))
    bm25_metrics = compute_metrics(questions, _ranked_for_metrics(bm25_records_, "rank", "score"))
    hybrid_metrics = compute_metrics(questions, _ranked_for_metrics(hybrid_records, "hybrid_rank", "rrf_score"))
    candidate_recall = _candidate_recall_report(questions, dense_records, bm25_records_, hybrid_records)
    case_studies = _case_studies(questions, dense_records, bm25_records_, hybrid_records)
    prompt_hash = _sha256_text(prompt_path.read_text(encoding="utf-8"))
    generation_packets = _generation_packets(hybrid_records, units_by_id, prompt_hash)
    public_bm25, public_hybrid, public_report = _public_sanity_report(repo_root, index, units_by_id)

    bm25_path = output_root / "bm25_results.jsonl"
    hybrid_path = output_root / "hybrid_results.jsonl"
    metrics_path = output_root / "metrics.json"
    generation_packets_path = output_root / "generation_packets.jsonl"
    public_bm25_path = output_root / "public_sanity_bm25_results.jsonl"
    public_hybrid_path = output_root / "public_sanity_hybrid_results.jsonl"
    public_report_path = output_root / "public_sanity_report.json"
    _write_jsonl(bm25_path, bm25_records_)
    _write_jsonl(hybrid_path, hybrid_records)
    _write_jsonl(generation_packets_path, generation_packets)
    _write_jsonl(public_bm25_path, public_bm25)
    _write_jsonl(public_hybrid_path, public_hybrid)
    _write_json(public_report_path, public_report)

    metrics_payload = {
        "phase": "7A",
        "definitions": dense_metrics["definitions"],
        "dense_baseline": dense_metrics,
        "bm25_only": bm25_metrics,
        "hybrid_rrf": hybrid_metrics,
        "comparison_table": _comparison_table(dense_metrics, bm25_metrics, hybrid_metrics),
        "candidate_recall": candidate_recall,
        "hard_negative_analysis": {
            "dense": dense_metrics["hard_negative_analysis"],
            "bm25": bm25_metrics["hard_negative_analysis"],
            "hybrid": hybrid_metrics["hard_negative_analysis"],
        },
        "case_studies": case_studies,
        "public_sanity": public_report,
    }
    _write_json(metrics_path, metrics_payload)

    output_paths = [
        bm25_path,
        hybrid_path,
        metrics_path,
        generation_packets_path,
        public_bm25_path,
        public_hybrid_path,
        public_report_path,
    ]
    manifest_path = repo_root / "data/processed/manifests/phase_7a_manifest.json"
    manifest = {
        "phase": "7A",
        "version": "1.0.0",
        "objective": "Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion",
        "corpus": {"retrieval_unit_count": len(units), "source": "Phase 4.5 retrieval units"},
        "dense_branch": {
            "backend": "Phase 5B Qdrant",
            "top_k": DENSE_TOP_K,
            "corpus_reembedded": False,
            "query_embeddings": "Phase 5A persisted benchmark query embeddings",
        },
        "bm25_branch": {
            "implementation": "local transparent BM25Okapi-style scorer",
            "top_k": BM25_TOP_K,
            "k1": BM25_K1,
            "b": BM25_B,
            "dependency": None,
            "tokenizer": {
                "lowercase": True,
                "unicode_word_extraction": True,
                "preserve_hyphenated_identifiers": True,
                "also_index_identifier_parts": True,
            },
            "documents_indexed": len(index.units),
            "unique_terms": len(index.document_frequencies),
        },
        "fusion": {"method": "Reciprocal Rank Fusion", "rrf_k": rrf_k},
        "counts": {
            "benchmark_questions": len(questions),
            "public_sanity_questions": len(PUBLIC_SANITY_QUESTIONS),
        },
        "input_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path)
            for path in [questions_path, units_path, phase5b_manifest_path, prompt_path]
            if path.exists()
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in output_paths
        },
        "metrics": metrics_payload["comparison_table"],
        "candidate_recall_summary": {
            branch: {
                cutoff: report["mean_recall_all_questions_with_gold"]
                for cutoff, report in branch_report.items()
            }
            for branch, branch_report in candidate_recall.items()
            if branch in {"dense", "bm25", "hybrid"}
        },
        "bm25_finds_gold_dense_misses": candidate_recall["bm25_finds_gold_dense_misses"],
        "case_studies": case_studies,
        "public_sanity": public_report,
        "generation_packet_policy": {
            "top_k": GENERATION_TOP_K,
            "uses_frozen_prompt": SYSTEM_PROMPT_VERSION,
            "exposes_dense_scores": False,
            "exposes_bm25_scores": False,
            "exposes_rrf_scores": False,
            "llm_generation_called": False,
        },
        "leakage_validation": validate_packets(generation_packets, units_by_id),
    }
    _write_json(manifest_path, manifest)

    return Phase7AResult(
        bm25_results=bm25_records_,
        hybrid_results=hybrid_records,
        metrics=metrics_payload,
        manifest=manifest,
        outputs=output_paths,
        manifest_path=manifest_path,
    )


__all__ = [
    "BM25_TOP_K",
    "DENSE_TOP_K",
    "RRF_K",
    "BM25Index",
    "bm25_search",
    "build_bm25_index",
    "fuse_dense_bm25",
    "rrf_score",
    "run_phase7a",
    "tokenize",
]

