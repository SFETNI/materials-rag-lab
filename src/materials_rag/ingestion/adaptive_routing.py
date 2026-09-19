"""Phase 9A adaptive retrieval routing over existing frozen retrieval strategies."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from materials_rag.ingestion.dense_retrieval import compute_metrics
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
from materials_rag.ingestion.multi_query_retrieval import repair_text_encoding
from materials_rag.ingestion.public_sanity import PUBLIC_SANITY_QUESTIONS
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

RouterStrategy = Literal["HYBRID", "MULTI_QUERY", "DECOMPOSITION"]
STRATEGIES: tuple[RouterStrategy, ...] = ("HYBRID", "MULTI_QUERY", "DECOMPOSITION")
ROUTER_PROMPT_VERSION = "adaptive_router_v1"
ROUTER_PROMPT_PATH = Path("docs/generation/adaptive_router_v1.md")
PHASE9A_ROOT = Path("data/processed/phase9a")
ROUTER_PACKETS_PATH = PHASE9A_ROOT / "router_packets.jsonl"
PUBLIC_ROUTER_PACKETS_PATH = PHASE9A_ROOT / "public_sanity_router_packets.jsonl"
ROUTER_DECISIONS_PATH = PHASE9A_ROOT / "router_decisions.jsonl"
PUBLIC_ROUTER_DECISIONS_PATH = PHASE9A_ROOT / "public_sanity_router_decisions.jsonl"
RAW_ROUTER_DECISIONS_PATH = PHASE9A_ROOT / "raw/router_decisions_raw.jsonl"
RAW_PUBLIC_ROUTER_DECISIONS_PATH = PHASE9A_ROOT / "raw/public_sanity_router_decisions_raw.jsonl"
ENCODING_NORMALIZATION_REPORT_PATH = PHASE9A_ROOT / "encoding_normalization_report.json"
BATCH_PROMPT_PATH = Path("experiments/19_adaptive_router_batch_prompt.txt")
KNOWN_CASE_IDS = ["SYNQ-001-A", "SYNQ-012-A", "SYNQ-002-A", "SYNQ-009-B", "SYNQ-014-B"]
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
    "retrieval_unit_ids",
    "scores",
    "metrics",
}


class RouterImportError(ValueError):
    """Raised when external isolated router output fails validation."""


class RouterFilesMissingError(FileNotFoundError):
    """Raised when Phase 9A router decisions are missing."""


@dataclass(frozen=True)
class RetrievalStrategy:
    name: RouterStrategy
    result_path: Path
    rank_key: str
    score_key: str
    config: dict[str, Any]

    def retrieve(self, records_by_id: dict[str, dict[str, Any]], question_id: str) -> dict[str, Any]:
        if question_id not in records_by_id:
            raise KeyError(f"{self.name} result missing question_id={question_id}")
        return records_by_id[question_id]


@dataclass
class Phase9AResult:
    status: str
    packets: list[dict[str, Any]]
    public_packets: list[dict[str, Any]]
    outputs: list[Path]
    metrics: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    missing_router_files: list[Path] | None = None


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


def build_router_packets(repo_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    questions = _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl")
    synthetic_packets = [_safe_packet(row["question_id"], row["query"]) for row in questions]
    public_packets = [_safe_packet(row["question_id"], row["query"]) for row in PUBLIC_SANITY_QUESTIONS]
    validate_safe_router_packets(synthetic_packets)
    validate_safe_router_packets(public_packets)
    return synthetic_packets, public_packets


def validate_safe_router_packets(packets: list[dict[str, Any]]) -> None:
    for packet in packets:
        if set(packet) != {"question_id", "original_question"}:
            raise ValueError(f"Unsafe router packet keys: {sorted(packet)}")
        if not packet["question_id"] or not packet["original_question"]:
            raise ValueError("Router packets require non-empty question_id and original_question.")
        if LEAKAGE_PACKET_KEYS & set(packet):
            raise ValueError(f"Router packet contains leakage keys: {LEAKAGE_PACKET_KEYS & set(packet)}")


def write_router_batch_prompt(
    repo_root: Path,
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> Path:
    prompt_text = (repo_root / ROUTER_PROMPT_PATH).read_text(encoding="utf-8").strip()
    lines = [
        prompt_text,
        "",
        "Route the following question-only requests. Return JSONL only, preserving input order.",
        "One JSON object per input line. No Markdown fences. No headings. No commentary.",
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


def _normalize_router_records(
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
        if isinstance(new_record.get("reason"), str):
            before = new_record["reason"]
            after, method = repair_text_encoding(before)
            if method:
                new_record["reason"] = after
                changes.append(
                    {
                        "question_id": question_id,
                        "field": "reason",
                        "before": before,
                        "after": after,
                        "repair_method": method,
                    }
                )
        normalized_records.append(new_record)
    _write_jsonl(output_path, normalized_records)
    return normalized_records, changes


def normalize_phase9a_router_inputs(
    repo_root: Path,
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> dict[str, Any]:
    router_path = repo_root / ROUTER_DECISIONS_PATH
    public_path = repo_root / PUBLIC_ROUTER_DECISIONS_PATH
    raw_router_path = repo_root / RAW_ROUTER_DECISIONS_PATH
    raw_public_path = repo_root / RAW_PUBLIC_ROUTER_DECISIONS_PATH
    _preserve_raw_file(router_path, raw_router_path)
    _preserve_raw_file(public_path, raw_public_path)
    changes: list[dict[str, Any]] = []
    if raw_router_path.exists():
        _, synthetic_changes = _normalize_router_records(raw_router_path, router_path, synthetic_packets)
        changes.extend(synthetic_changes)
    if raw_public_path.exists():
        _, public_changes = _normalize_router_records(raw_public_path, public_path, public_packets)
        changes.extend(public_changes)
    raw_paths = [path for path in [raw_router_path, raw_public_path] if path.exists()]
    normalized_paths = [path for path in [router_path, public_path] if path.exists()]
    report = {
        "phase": "9A",
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


def validate_router_decisions(
    decision_path: Path,
    expected_packets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not decision_path.exists():
        raise RouterFilesMissingError(f"Missing external router file: {decision_path}")
    records = _read_jsonl(decision_path)
    expected_ids = [packet["question_id"] for packet in expected_packets]
    actual_ids = [record.get("question_id") for record in records]
    if len(actual_ids) != len(set(actual_ids)):
        duplicates = sorted({question_id for question_id in actual_ids if actual_ids.count(question_id) > 1})
        raise RouterImportError(f"Duplicate router question IDs: {duplicates}")
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise RouterImportError(
            "Router IDs must match expected packet order exactly. "
            f"missing={missing}; unexpected={unexpected}."
        )
    expected_by_id = {packet["question_id"]: packet for packet in expected_packets}
    validated: list[dict[str, Any]] = []
    for record in records:
        if set(record) != {"question_id", "original_question", "strategy", "reason"}:
            raise RouterImportError(f"{record.get('question_id')} has invalid keys: {sorted(record)}")
        question_id = record["question_id"]
        expected_question = expected_by_id[question_id]["original_question"]
        if record["original_question"] != expected_question:
            raise RouterImportError(f"{question_id} original_question does not match frozen text.")
        strategy = record["strategy"]
        if strategy not in STRATEGIES:
            raise RouterImportError(f"{question_id} invalid strategy: {strategy}")
        reason = record["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise RouterImportError(f"{question_id} reason must be a non-empty string.")
        validated.append(
            {
                "question_id": question_id,
                "original_question": expected_question,
                "strategy": strategy,
                "reason": reason.strip(),
            }
        )
    return validated


def strategy_registry() -> dict[RouterStrategy, RetrievalStrategy]:
    return {
        "HYBRID": RetrievalStrategy(
            name="HYBRID",
            result_path=Path("data/processed/phase7a/hybrid_results.jsonl"),
            rank_key="hybrid_rank",
            score_key="rrf_score",
            config={"source_phase": "7A", "retrieval": "Dense + BM25 + RRF"},
        ),
        "MULTI_QUERY": RetrievalStrategy(
            name="MULTI_QUERY",
            result_path=Path("data/processed/phase8a/multi_query_results.jsonl"),
            rank_key="multi_query_rank",
            score_key="multi_query_rrf_score",
            config={"source_phase": "8A", "retrieval": "Multi-query Hybrid RRF"},
        ),
        "DECOMPOSITION": RetrievalStrategy(
            name="DECOMPOSITION",
            result_path=Path("data/processed/phase8b/decomposition_results.jsonl"),
            rank_key="decomposition_rank",
            score_key="decomposition_rrf_score",
            config={"source_phase": "8B", "retrieval": "Decomposition Hybrid RRF"},
        ),
    }


def _load_strategy_records(repo_root: Path, registry: dict[RouterStrategy, RetrievalStrategy]) -> dict[RouterStrategy, dict[str, dict[str, Any]]]:
    loaded: dict[RouterStrategy, dict[str, dict[str, Any]]] = {}
    for name, strategy in registry.items():
        path = repo_root / strategy.result_path
        if not path.exists():
            raise FileNotFoundError(f"Required frozen strategy result missing: {strategy.result_path}")
        rows = _read_jsonl(path)
        loaded[name] = {row["question_id"]: row for row in rows}
    return loaded


def _verify_strategy_manifests(repo_root: Path) -> dict[str, Any]:
    required = {
        "HYBRID": repo_root / "data/processed/manifests/phase_7a_manifest.json",
        "MULTI_QUERY": repo_root / "data/processed/manifests/phase_8a_manifest.json",
        "DECOMPOSITION": repo_root / "data/processed/manifests/phase_8b_manifest.json",
    }
    report = {}
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required manifest missing for {name}: {_relative(path, repo_root)}")
        data = _read_json(path)
        status = data.get("status", "complete")
        if status != "complete":
            raise ValueError(f"Manifest for {name} is not complete: status={status}")
        report[name] = {"path": _relative(path, repo_root), "sha256": sha256_for_file(path)}
    return report


def _result_score(result: dict[str, Any], score_key: str) -> float:
    value = result.get(score_key)
    return float(value) if value is not None else 0.0


def _normalize_selected_results(row: dict[str, Any], strategy: RetrievalStrategy) -> list[dict[str, Any]]:
    normalized = []
    for result in sorted(row["results"], key=lambda item: item[strategy.rank_key]):
        payload = {
            "retrieval_unit_id": result["retrieval_unit_id"],
            "adaptive_rank": int(result[strategy.rank_key]),
            "adaptive_score": _result_score(result, strategy.score_key),
            "selected_strategy": strategy.name,
            "source_rank_key": strategy.rank_key,
            "source_score_key": strategy.score_key,
            "kind": result.get("kind"),
            "source": result.get("source"),
            "evidence_id": result.get("evidence_id"),
            "document_id": result.get("document_id"),
        }
        normalized.append(payload)
    return normalized


def _trace_event(
    event_type: str,
    actor: str,
    input_ids: list[str] | None = None,
    output_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "actor": actor,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "input_ids": input_ids or [],
        "output_ids": output_ids or [],
        "metadata": metadata or {},
    }


def _build_adaptive_record(
    decision: dict[str, Any],
    strategy_records: dict[RouterStrategy, dict[str, dict[str, Any]]],
    registry: dict[RouterStrategy, RetrievalStrategy],
) -> dict[str, Any]:
    strategy_name: RouterStrategy = decision["strategy"]
    strategy = registry[strategy_name]
    frozen_row = strategy.retrieve(strategy_records[strategy_name], decision["question_id"])
    results = _normalize_selected_results(frozen_row, strategy)
    trace = [
        _trace_event(
            "QUESTION_RECEIVED",
            "phase9a_controller",
            input_ids=[decision["question_id"]],
            metadata={"question_id": decision["question_id"]},
        ),
        _trace_event(
            "ROUTE_SELECTED",
            "isolated_router",
            input_ids=[decision["question_id"]],
            output_ids=[strategy_name],
            metadata={"strategy": strategy_name, "router_reason": decision["reason"]},
        ),
        _trace_event(
            "RETRIEVAL_COMPLETED",
            "phase9a_strategy_registry",
            input_ids=[decision["question_id"], strategy_name],
            output_ids=[result["retrieval_unit_id"] for result in results[:5]],
            metadata={"selected_strategy": strategy_name, "result_count": len(results)},
        ),
    ]
    return {
        "question_id": decision["question_id"],
        "query": decision["original_question"],
        "selected_strategy": strategy_name,
        "router_reason": decision["reason"],
        "strategy_provenance": dict(strategy.config),
        "results": results,
        "trace": trace,
    }


def _build_adaptive_records(
    decisions: list[dict[str, Any]],
    strategy_records: dict[RouterStrategy, dict[str, dict[str, Any]]],
    registry: dict[RouterStrategy, RetrievalStrategy],
) -> list[dict[str, Any]]:
    return [_build_adaptive_record(decision, strategy_records, registry) for decision in decisions]


def _ranked_for_metrics(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["question_id"]: [
            {"retrieval_unit_id": result["retrieval_unit_id"], "score": result["adaptive_score"]}
            for result in sorted(row["results"], key=lambda item: item["adaptive_rank"])
        ]
        for row in records
    }


def _system_ranked_for_metrics(strategy_rows: dict[str, dict[str, Any]], rank_key: str, score_key: str) -> dict[str, list[dict[str, Any]]]:
    return {
        question_id: [
            {"retrieval_unit_id": result["retrieval_unit_id"], "score": _result_score(result, score_key)}
            for result in sorted(row["results"], key=lambda item: item[rank_key])
        ]
        for question_id, row in strategy_rows.items()
    }


def _comparison_table(
    phase8b_metrics: dict[str, Any], adaptive_metrics: dict[str, Any]
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
    decomp_by_metric = {
        row["metric"]: row["decomposition_hybrid"] for row in phase8b_metrics["comparison_table"]
    }
    return [
        {
            "metric": metric,
            "dense": phase8b_metrics["dense"]["overall_non_abstain"][metric],
            "dense_cross_encoder": phase8b_metrics["dense_cross_encoder"]["overall_non_abstain"][metric],
            "hybrid_rrf": phase8b_metrics["hybrid_rrf"]["overall_non_abstain"][metric],
            "hybrid_rrf_cross_encoder": phase8b_metrics["hybrid_rrf_cross_encoder"][
                "overall_non_abstain"
            ][metric],
            "multi_query_hybrid": phase8b_metrics["multi_query_hybrid"]["overall_non_abstain"][metric],
            "decomposition_hybrid": decomp_by_metric[metric],
            "adaptive_router": adaptive_metrics["overall_non_abstain"][metric],
        }
        for metric in ordered
    ]


def _routing_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    query_costs = {"HYBRID": 1, "MULTI_QUERY": 4}
    counts = {strategy: 0 for strategy in STRATEGIES}
    total_queries = 0
    for row in records:
        strategy = row["selected_strategy"]
        counts[strategy] += 1
        if strategy == "DECOMPOSITION":
            total_queries += int(row["strategy_provenance"].get("retrieval_query_count", 0) or 0)
        else:
            total_queries += query_costs[strategy]
    # DECOMPOSITION records get precise costs patched by _attach_decomposition_costs.
    question_count = len(records)
    return {
        "question_count": question_count,
        "counts": counts,
        "percentages": {strategy: counts[strategy] / question_count if question_count else 0.0 for strategy in STRATEGIES},
        "total_retrieval_queries_implied": total_queries,
        "mean_retrieval_queries_per_original_question": total_queries / question_count if question_count else 0.0,
    }


def _attach_decomposition_costs(records: list[dict[str, Any]], decomp_rows: dict[str, dict[str, Any]]) -> None:
    for row in records:
        if row["selected_strategy"] == "DECOMPOSITION":
            count = decomp_rows[row["question_id"]].get("retrieval_query_count", 1)
            row["strategy_provenance"]["retrieval_query_count"] = count
        elif row["selected_strategy"] == "MULTI_QUERY":
            row["strategy_provenance"]["retrieval_query_count"] = 4
        else:
            row["strategy_provenance"]["retrieval_query_count"] = 1


def _question_metrics_by_strategy(
    questions: list[dict[str, Any]],
    strategy_ranked: dict[RouterStrategy, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for question in questions:
        qid = question["question_id"]
        metrics[qid] = {}
        for strategy, ranked_by_question in strategy_ranked.items():
            single = compute_metrics([question], {qid: ranked_by_question[qid]})
            metrics[qid][strategy] = single["per_question"][0]["metrics"]
    return metrics


def _oracle_analysis(
    questions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    strategy_records: dict[RouterStrategy, dict[str, dict[str, Any]]],
    registry: dict[RouterStrategy, RetrievalStrategy],
) -> dict[str, Any]:
    strategy_ranked = {
        strategy: _system_ranked_for_metrics(rows, registry[strategy].rank_key, registry[strategy].score_key)
        for strategy, rows in strategy_records.items()
        if all(question["question_id"] in rows for question in questions)
    }
    per_question_metrics = _question_metrics_by_strategy(questions, strategy_ranked)
    decisions_by_id = {row["question_id"]: row for row in decisions}
    rows = []
    for question in questions:
        qid = question["question_id"]
        best_strategy = max(
            STRATEGIES,
            key=lambda strategy: (
                per_question_metrics[qid][strategy]["ndcg@10"],
                per_question_metrics[qid][strategy]["mrr"],
                per_question_metrics[qid][strategy]["recall@10"],
            ),
        )
        selected = decisions_by_id[qid]["strategy"]
        selected_metric = per_question_metrics[qid][selected]["ndcg@10"]
        best_metric = per_question_metrics[qid][best_strategy]["ndcg@10"]
        rows.append(
            {
                "question_id": qid,
                "selected_strategy": selected,
                "best_retrospective_strategy": best_strategy,
                "oracle_metric": "ndcg@10_then_mrr_then_recall@10",
                "selected_ndcg@10": selected_metric,
                "best_ndcg@10": best_metric,
                "metric_delta": best_metric - selected_metric,
                "question_structure": decisions_by_id[qid]["reason"],
            }
        )
    matches = sum(row["selected_strategy"] == row["best_retrospective_strategy"] for row in rows)
    return {
        "label": "EVALUATION-ONLY ORACLE",
        "selection_metric": "ndcg@10_then_mrr_then_recall@10",
        "agreement_count": matches,
        "question_count": len(rows),
        "agreement_rate": matches / len(rows) if rows else 0.0,
        "per_question": rows,
    }


def _public_strategy_registry() -> dict[RouterStrategy, RetrievalStrategy]:
    return {
        "HYBRID": RetrievalStrategy(
            name="HYBRID",
            result_path=Path("data/processed/phase7a/public_sanity_hybrid_results.jsonl"),
            rank_key="hybrid_rank",
            score_key="rrf_score",
            config={"source_phase": "7A", "retrieval": "Dense + BM25 + RRF"},
        ),
        "MULTI_QUERY": RetrievalStrategy(
            name="MULTI_QUERY",
            result_path=Path("data/processed/phase8a/public_sanity_multi_query_results.jsonl"),
            rank_key="multi_query_rank",
            score_key="multi_query_rrf_score",
            config={"source_phase": "8A", "retrieval": "Multi-query Hybrid RRF"},
        ),
        "DECOMPOSITION": RetrievalStrategy(
            name="DECOMPOSITION",
            result_path=Path("data/processed/phase8b/public_sanity_decomposition_results.jsonl"),
            rank_key="decomposition_rank",
            score_key="decomposition_rrf_score",
            config={"source_phase": "8B", "retrieval": "Decomposition Hybrid RRF"},
        ),
    }


def _packet_for_generation(
    row: dict[str, Any], units_by_id: dict[str, dict[str, Any]], prompt_hash: str
) -> dict[str, Any]:
    ordered = sorted(row["results"], key=lambda result: result["adaptive_rank"])[:GENERATION_TOP_K]
    retrieval_unit_ids = [result["retrieval_unit_id"] for result in ordered]
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in retrieval_unit_ids]
    generator_input = format_generator_input(row["query"], context_items)
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "top_k": GENERATION_TOP_K,
        "retrieval_unit_ids": retrieval_unit_ids,
        "retrieval_backend": "Phase 9A Adaptive Router",
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
        raise ValueError(f"Phase 9A generation packet validation failed: {validation}")
    forbidden = [
        "router",
        "selected_strategy",
        "oracle",
        "hybrid_rank",
        "multi_query",
        "decomposition_rank",
        "adaptive_rank",
        "score:",
        "gold",
        "hard_negative",
    ]
    for packet in packets:
        lowered = packet["generator_input"].lower()
        for term in forbidden:
            if term in lowered:
                raise ValueError(f"Phase 9A packet exposes runtime/evaluation internals: {term}")
    return packets


def _expected_rank(row: dict[str, Any], expected_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_ids)
    ranked = sorted(row["results"], key=lambda result: result["adaptive_rank"])
    matches = [result for result in ranked if result["retrieval_unit_id"] in expected]
    best = matches[0] if matches else None
    return {
        "best_expected_adaptive_rank": best.get("adaptive_rank") if best else None,
        "best_expected_unit_id": best.get("retrieval_unit_id") if best else None,
        "top5_ids": [result["retrieval_unit_id"] for result in ranked[:5]],
    }


def _public_sanity_report(records: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in records}
    decisions_by_id = {row["question_id"]: row for row in decisions}
    rows = []
    for question in PUBLIC_SANITY_QUESTIONS:
        qid = question["question_id"]
        rank_info = _expected_rank(by_question[qid], question["expected_retrieval_unit_ids"])
        rows.append(
            {
                "question_id": qid,
                "query": question["query"],
                "source_type": question["source_type"],
                "selected_strategy": decisions_by_id[qid]["strategy"],
                "router_reason": decisions_by_id[qid]["reason"],
                "expected_retrieval_unit_ids": question["expected_retrieval_unit_ids"],
                **rank_info,
            }
        )
    return {
        "question_count": len(rows),
        "strategy_counts": {strategy: sum(row["selected_strategy"] == strategy for row in rows) for strategy in STRATEGIES},
        "pubsan_005": next(row for row in rows if row["question_id"] == "PUBSAN-005"),
        "per_question": rows,
    }


def _rank_for_unit(row: dict[str, Any], unit_id: str) -> int | None:
    for result in row["results"]:
        if result["retrieval_unit_id"] == unit_id:
            return result.get("adaptive_rank")
    return None


def _rank_for_evidence(row: dict[str, Any], evidence_id: str) -> int | None:
    for result in row["results"]:
        if result.get("evidence_id") == evidence_id:
            return result.get("adaptive_rank")
    return None


def _strategy_rank_for_evidence(
    strategy_row: dict[str, Any], rank_key: str, evidence_or_unit_id: str
) -> int | None:
    for result in strategy_row["results"]:
        if result.get("evidence_id") == evidence_or_unit_id or result["retrieval_unit_id"] == evidence_or_unit_id:
            return result.get(rank_key)
    return None


def _known_case_report(
    adaptive_records: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    strategy_records: dict[RouterStrategy, dict[str, dict[str, Any]]],
    registry: dict[RouterStrategy, RetrievalStrategy],
) -> dict[str, Any]:
    adaptive_by_id = {row["question_id"]: row for row in adaptive_records}
    decisions_by_id = {row["question_id"]: row for row in decisions}
    tracked = {
        "SYNQ-001-A": ["FAT-B017#results", "FAT-B017#conditions"],
        "SYNQ-012-A": ["RCA-B017-004-rev2#status", "RCA-B017-004-rev2#conclusion"],
        "SYNQ-002-A": [
            "chunk|document_mrl_internal_in718_v1_FAT-B017|results|0",
            "chunk|document_mrl_internal_in718_v1_FAT-B017|assessment|0",
            "chunk|document_mrl_internal_in718_v1_MAT-SPEC-IN718-001|outcomes|0",
        ],
        "SYNQ-009-B": [
            "chunk|document_mrl_internal_in718_v1_FAT-B021|results|0",
            "chunk|document_mrl_internal_in718_v1_MAT-SPEC-IN718-001|acceptance|0",
            "chunk|document_mrl_internal_in718_v1_FAT-B021|assessment|0",
        ],
        "SYNQ-014-B": [
            "chunk|document_mrl_internal_in718_v1_BUILD-B017|events|0",
            "chunk|document_mrl_internal_in718_v1_FRACT-B017|interpretation|0",
            "chunk|document_mrl_internal_in718_v1_RCA-B017-004-rev2|conclusion|0",
        ],
    }
    report = {}
    for qid in KNOWN_CASE_IDS:
        decision = decisions_by_id[qid]
        evidence_report = {}
        for item_id in tracked[qid]:
            ranks = {}
            for strategy in STRATEGIES:
                ranks[strategy] = _strategy_rank_for_evidence(
                    strategy_records[strategy][qid], registry[strategy].rank_key, item_id
                )
            evidence_report[item_id] = {
                "adaptive_rank": _rank_for_evidence(adaptive_by_id[qid], item_id)
                if "#" in item_id and not item_id.startswith("chunk|")
                else _rank_for_unit(adaptive_by_id[qid], item_id),
                "strategy_ranks": ranks,
            }
        report[qid] = {
            "selected_strategy": decision["strategy"],
            "router_reason": decision["reason"],
            "evidence": evidence_report,
        }
    return report


def _trace_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = ["gold", "hard_negative", "answerability", "split", "reference_answer"]
    violations = []
    for row in records:
        text = json.dumps(row.get("trace", []), ensure_ascii=False).lower()
        hits = [term for term in forbidden if term in text]
        if hits:
            violations.append({"question_id": row["question_id"], "terms": hits})
    return {"passed": not violations, "violations": violations}


def _write_waiting_manifest(
    repo_root: Path,
    setup_outputs: list[Path],
    missing: list[Path],
    synthetic_packets: list[dict[str, str]],
    public_packets: list[dict[str, str]],
) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / "data/processed/manifests/phase_9a_manifest.json"
    manifest = {
        "phase": "9A",
        "version": "1.0.0",
        "status": "waiting_for_external_router_decisions",
        "objective": "Adaptive routing infrastructure over existing retrieval strategies",
        "strategy_registry": {name: strategy.config for name, strategy in strategy_registry().items()},
        "packet_counts": {"synthetic": len(synthetic_packets), "public_sanity": len(public_packets)},
        "missing_router_files": [_relative(path, repo_root) for path in missing],
        "router_prompt": {
            "version": ROUTER_PROMPT_VERSION,
            "path": ROUTER_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / ROUTER_PROMPT_PATH),
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in setup_outputs
        },
    }
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def run_phase9a(repo_root: Path | None = None) -> Phase9AResult:
    """Prepare safe router inputs and run Phase 9A when external decisions exist."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / PHASE9A_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    synthetic_packets, public_packets = build_router_packets(repo_root)
    packets_path = repo_root / ROUTER_PACKETS_PATH
    public_packets_path = repo_root / PUBLIC_ROUTER_PACKETS_PATH
    _write_jsonl(packets_path, synthetic_packets)
    _write_jsonl(public_packets_path, public_packets)
    batch_prompt_path = write_router_batch_prompt(repo_root, synthetic_packets, public_packets)

    router_paths = [repo_root / ROUTER_DECISIONS_PATH, repo_root / PUBLIC_ROUTER_DECISIONS_PATH]
    missing = [path for path in router_paths if not path.exists()]
    setup_outputs = [packets_path, public_packets_path, batch_prompt_path]
    if missing:
        manifest, manifest_path = _write_waiting_manifest(
            repo_root, setup_outputs, missing, synthetic_packets, public_packets
        )
        return Phase9AResult(
            status="waiting_for_external_router_decisions",
            packets=synthetic_packets,
            public_packets=public_packets,
            outputs=setup_outputs,
            manifest=manifest,
            manifest_path=manifest_path,
            missing_router_files=missing,
        )

    normalization_report = normalize_phase9a_router_inputs(repo_root, synthetic_packets, public_packets)
    decisions = validate_router_decisions(repo_root / ROUTER_DECISIONS_PATH, synthetic_packets)
    public_decisions = validate_router_decisions(repo_root / PUBLIC_ROUTER_DECISIONS_PATH, public_packets)

    registry = strategy_registry()
    public_registry = _public_strategy_registry()
    manifest_inputs = _verify_strategy_manifests(repo_root)
    strategy_records = _load_strategy_records(repo_root, registry)
    public_strategy_records = _load_strategy_records(repo_root, public_registry)

    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    phase8b_metrics_path = repo_root / "data/processed/phase8b/metrics.json"
    generator_prompt_path = repo_root / GENERATOR_PROMPT_PATH
    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    units_by_id = {unit["id"]: unit for unit in units}
    phase8b_metrics = _read_json(phase8b_metrics_path)

    adaptive_records = _build_adaptive_records(decisions, strategy_records, registry)
    public_adaptive_records = _build_adaptive_records(public_decisions, public_strategy_records, public_registry)
    _attach_decomposition_costs(adaptive_records, strategy_records["DECOMPOSITION"])
    _attach_decomposition_costs(public_adaptive_records, public_strategy_records["DECOMPOSITION"])

    adaptive_metrics = compute_metrics(questions, _ranked_for_metrics(adaptive_records))
    routing_diagnostics = _routing_diagnostics(adaptive_records)
    oracle = _oracle_analysis(questions, decisions, strategy_records, registry)
    public_report = _public_sanity_report(public_adaptive_records, public_decisions)
    known_cases = _known_case_report(adaptive_records, decisions, strategy_records, registry)
    generator_prompt_hash = _sha256_text(generator_prompt_path.read_text(encoding="utf-8"))
    generation_packets = _generation_packets(adaptive_records, units_by_id, generator_prompt_hash)
    trace_validation = _trace_validation(adaptive_records)
    if not trace_validation["passed"]:
        raise ValueError(f"Phase 9A trace leakage validation failed: {trace_validation}")

    metrics_payload = {
        "phase": "9A",
        "definitions": adaptive_metrics["definitions"],
        "dense": phase8b_metrics["dense"],
        "dense_cross_encoder": phase8b_metrics["dense_cross_encoder"],
        "hybrid_rrf": phase8b_metrics["hybrid_rrf"],
        "hybrid_rrf_cross_encoder": phase8b_metrics["hybrid_rrf_cross_encoder"],
        "multi_query_hybrid": phase8b_metrics["multi_query_hybrid"],
        "decomposition_hybrid": phase8b_metrics["decomposition_hybrid"],
        "adaptive_router": adaptive_metrics,
        "comparison_table": _comparison_table(phase8b_metrics, adaptive_metrics),
        "dev_metrics": adaptive_metrics["by_split"].get("dev", {}),
        "challenge_metrics": adaptive_metrics["by_split"].get("challenge", {}),
        "routing_diagnostics": routing_diagnostics,
        "hard_negative_analysis": adaptive_metrics["hard_negative_analysis"],
        "evaluation_only_oracle": oracle,
        "router_confusion_analysis": oracle["per_question"],
        "known_cases": known_cases,
        "public_sanity": public_report,
    }

    results_path = output_root / "adaptive_results.jsonl"
    metrics_path = output_root / "metrics.json"
    packets_out_path = output_root / "generation_packets.jsonl"
    public_results_path = output_root / "public_sanity_adaptive_results.jsonl"
    public_report_path = output_root / "public_sanity_report.json"
    _write_jsonl(results_path, adaptive_records)
    _write_json(metrics_path, metrics_payload)
    _write_jsonl(packets_out_path, generation_packets)
    _write_jsonl(public_results_path, public_adaptive_records)
    _write_json(public_report_path, public_report)

    manifest_path = repo_root / "data/processed/manifests/phase_9a_manifest.json"
    output_paths = [
        *setup_outputs,
        repo_root / ENCODING_NORMALIZATION_REPORT_PATH,
        results_path,
        metrics_path,
        packets_out_path,
        public_results_path,
        public_report_path,
    ]
    input_paths = [
        questions_path,
        units_path,
        phase8b_metrics_path,
        repo_root / ROUTER_DECISIONS_PATH,
        repo_root / PUBLIC_ROUTER_DECISIONS_PATH,
        repo_root / RAW_ROUTER_DECISIONS_PATH,
        repo_root / RAW_PUBLIC_ROUTER_DECISIONS_PATH,
        generator_prompt_path,
    ]
    input_paths.extend(repo_root / strategy.result_path for strategy in registry.values())
    input_paths.extend(repo_root / strategy.result_path for strategy in public_registry.values())
    manifest = {
        "phase": "9A",
        "version": "1.0.0",
        "status": "complete",
        "objective": "Adaptive routing over existing frozen retrieval strategies",
        "router_prompt": {
            "version": ROUTER_PROMPT_VERSION,
            "path": ROUTER_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / ROUTER_PROMPT_PATH),
        },
        "configuration": {
            "available_strategies": list(STRATEGIES),
            "new_retrieval_algorithms_added": False,
            "answer_generator_called": False,
            "router_decisions_external": True,
        },
        "strategy_registry": {name: strategy.config for name, strategy in registry.items()},
        "verified_strategy_manifests": manifest_inputs,
        "input_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in input_paths if path.exists()
        },
        "output_artifact_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in output_paths
        },
        "encoding_normalization": normalization_report,
        "metrics": metrics_payload["comparison_table"],
        "dev_metrics": metrics_payload["dev_metrics"],
        "challenge_metrics": metrics_payload["challenge_metrics"],
        "routing_diagnostics": routing_diagnostics,
        "evaluation_only_oracle_summary": {
            key: value for key, value in oracle.items() if key != "per_question"
        },
        "known_cases": known_cases,
        "public_sanity": public_report,
        "leakage_validation": validate_packets(generation_packets, units_by_id),
        "trace_validation": trace_validation,
    }
    _write_json(manifest_path, manifest)

    return Phase9AResult(
        status="complete",
        packets=synthetic_packets,
        public_packets=public_packets,
        outputs=output_paths,
        metrics=metrics_payload,
        manifest=manifest,
        manifest_path=manifest_path,
    )


__all__ = [
    "STRATEGIES",
    "RetrievalStrategy",
    "RouterFilesMissingError",
    "RouterImportError",
    "build_router_packets",
    "normalize_phase9a_router_inputs",
    "run_phase9a",
    "strategy_registry",
    "validate_router_decisions",
    "validate_safe_router_packets",
]
