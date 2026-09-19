"""Phase 9B bounded corrective retrieval orchestration scaffolding."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from materials_rag.ingestion.adaptive_routing import (
    STRATEGIES,
    RouterStrategy,
    _load_strategy_records,
    _public_strategy_registry,
    _trace_event,
    run_phase9a,
    strategy_registry,
)
from materials_rag.ingestion.dense_retrieval import compute_metrics
from materials_rag.ingestion.generation_packets import (
    PROMPT_PATH as GENERATOR_PROMPT_PATH,
)
from materials_rag.ingestion.generation_packets import (
    SYSTEM_PROMPT_VERSION,
    _context_item,
    _packet_hash,
    format_generator_input,
    validate_packets,
)
from materials_rag.ingestion.multi_query_retrieval import repair_text_encoding
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

ASSESSMENT_CONTEXT_K = 10
FINAL_CONTEXT_K = 5
ASSESSOR_PROMPT_VERSION = "evidence_assessor_initial_v2"
ASSESSOR_PROMPT_PATH = Path("docs/generation/evidence_assessor_initial_v2.md")
FINAL_ASSESSOR_PROMPT_VERSION = "evidence_assessor_final_v1"
FINAL_ASSESSOR_PROMPT_PATH = Path("docs/generation/evidence_assessor_final_v1.md")
PHASE9B_ROOT = Path("data/processed/phase9b")
INITIAL_PACKETS_PATH = PHASE9B_ROOT / "initial_assessment_packets.jsonl"
PUBLIC_INITIAL_PACKETS_PATH = PHASE9B_ROOT / "public_sanity_initial_assessment_packets.jsonl"
INITIAL_ASSESSMENTS_PATH = PHASE9B_ROOT / "initial_assessments.jsonl"
PUBLIC_INITIAL_ASSESSMENTS_PATH = PHASE9B_ROOT / "public_sanity_initial_assessments.jsonl"
RAW_INITIAL_ASSESSMENTS_PATH = PHASE9B_ROOT / "raw/initial_assessments_raw.jsonl"
RAW_PUBLIC_INITIAL_ASSESSMENTS_PATH = PHASE9B_ROOT / "raw/public_sanity_initial_assessments_raw.jsonl"
RAW_POST_ASSESSMENTS_PATH = PHASE9B_ROOT / "raw/post_correction_assessments_raw.jsonl"
RAW_PUBLIC_POST_ASSESSMENTS_PATH = PHASE9B_ROOT / "raw/public_sanity_post_correction_assessments_raw.jsonl"
ENCODING_NORMALIZATION_REPORT_PATH = PHASE9B_ROOT / "encoding_normalization_report.json"
INITIAL_BATCH_PROMPT_PATH = Path("experiments/20_evidence_assessor_batch_prompt.txt")
POST_BATCH_PROMPT_PATH = Path("experiments/20b_post_correction_assessor_batch_prompt.txt")
POST_PACKETS_PATH = PHASE9B_ROOT / "post_correction_assessment_packets.jsonl"
PUBLIC_POST_PACKETS_PATH = PHASE9B_ROOT / "public_sanity_post_correction_assessment_packets.jsonl"
POST_ASSESSMENTS_PATH = PHASE9B_ROOT / "post_correction_assessments.jsonl"
PUBLIC_POST_ASSESSMENTS_PATH = PHASE9B_ROOT / "public_sanity_post_correction_assessments.jsonl"
MANIFEST_PATH = Path("data/processed/manifests/phase_9b_manifest.json")
FINAL_SINGLE_PROMPT_DIR = Path("experiments/phase9b_assessor_final_v1")

EvidenceStatus = Literal["SUPPORTED", "PARTIAL", "MISSING", "CONFLICTING", "SUPERSEDED"]
InitialOverallStatus = Literal["SUFFICIENT", "NEEDS_CORRECTION"]
FinalOverallStatus = Literal["SUFFICIENT", "INSUFFICIENT_WITH_RESIDUAL"]
CorrectionStrategy = Literal["NONE", "HYBRID", "MULTI_QUERY", "DECOMPOSITION"]
TraceEventType = Literal[
    "QUESTION_RECEIVED",
    "ROUTE_SELECTED",
    "RETRIEVAL_COMPLETED",
    "EVIDENCE_ASSESSED",
    "CORRECTION_SELECTED",
    "CORRECTIVE_RETRIEVAL_COMPLETED",
    "FINAL_EVIDENCE_ASSESSED",
]

EVIDENCE_STATUSES = {"SUPPORTED", "PARTIAL", "MISSING", "CONFLICTING", "SUPERSEDED"}
INITIAL_OVERALL_STATUSES = {"SUFFICIENT", "NEEDS_CORRECTION"}
FINAL_OVERALL_STATUSES = {"SUFFICIENT", "INSUFFICIENT_WITH_RESIDUAL"}
CORRECTION_STRATEGIES = {"NONE", *STRATEGIES}
TRACE_EVENT_TYPES = {
    "QUESTION_RECEIVED",
    "ROUTE_SELECTED",
    "RETRIEVAL_COMPLETED",
    "EVIDENCE_ASSESSED",
    "CORRECTION_SELECTED",
    "CORRECTIVE_RETRIEVAL_COMPLETED",
    "FINAL_EVIDENCE_ASSESSED",
}
LEAKAGE_TERMS = [
    "gold_chunk_ids",
    "gold_evidence_ids",
    "hard_negative",
    "reference_answer",
    "answerability",
    "benchmark split",
    "oracle",
    "metric delta",
]


class EvidenceAssessmentImportError(ValueError):
    """Raised when external Evidence Assessor output fails validation."""


@dataclass(frozen=True)
class RetrievalTool:
    name: RouterStrategy
    source_phase: str
    result_path: Path
    rank_key: str
    description: str


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    description: str
    status: EvidenceStatus
    supporting_context_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorrectionDecision:
    corrective_retrieval_performed: bool
    correction_strategy: CorrectionStrategy
    reason: str = ""


@dataclass(frozen=True)
class EvidenceLedger:
    question_id: str
    assessment_stage: str
    evidence_requirements: tuple[EvidenceRequirement, ...]
    overall_status: str
    missing_evidence_summary: str
    selected_context_ids: tuple[str, ...]
    correction_strategy: CorrectionStrategy = "NONE"


@dataclass(frozen=True)
class InitialEvidenceAssessment:
    question_id: str
    assessment_stage: Literal["INITIAL"]
    overall_status: InitialOverallStatus
    correction_strategy: CorrectionStrategy


@dataclass(frozen=True)
class FinalEvidenceAssessment:
    question_id: str
    assessment_stage: Literal["FINAL"]
    overall_status: FinalOverallStatus
    correction_strategy: Literal["NONE"]


@dataclass(frozen=True)
class EvidenceAssessment:
    question_id: str
    ledger: EvidenceLedger
    current_strategy: RouterStrategy
    correction_decision: CorrectionDecision


@dataclass(frozen=True)
class TraceEvent:
    event_type: TraceEventType
    actor: str
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    timestamp_utc: str
    metadata: dict[str, Any]


@dataclass
class Phase9BResult:
    status: str
    initial_packets: list[dict[str, Any]]
    public_initial_packets: list[dict[str, Any]]
    outputs: list[Path]
    manifest: dict[str, Any]
    manifest_path: Path
    missing_assessment_files: list[Path] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _json_text(record: Any) -> str:
    return json.dumps(record, ensure_ascii=False).lower()


def retrieval_tool_registry() -> dict[RouterStrategy, RetrievalTool]:
    """Expose Phase 7A/8A/8B frozen outputs as simple retrieval tools."""

    registry = strategy_registry()
    return {
        name: RetrievalTool(
            name=name,
            source_phase=strategy.config["source_phase"],
            result_path=strategy.result_path,
            rank_key=strategy.rank_key,
            description=strategy.config["retrieval"],
        )
        for name, strategy in registry.items()
    }


def _context_from_unit(unit: dict[str, Any]) -> dict[str, Any]:
    item = _context_item(unit)
    item["context_id"] = item.pop("id")
    return item


def _safe_evidence_bundle(
    adaptive_record: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    limit: int = ASSESSMENT_CONTEXT_K,
) -> list[dict[str, Any]]:
    bundle: list[dict[str, Any]] = []
    for result in sorted(adaptive_record["results"], key=lambda row: row["adaptive_rank"])[:limit]:
        unit_id = result["retrieval_unit_id"]
        bundle.append(_context_from_unit(units_by_id[unit_id]))
    return bundle


def _assessment_packet(
    adaptive_record: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question_id": adaptive_record["question_id"],
        "original_question": adaptive_record["query"],
        "current_strategy": adaptive_record["selected_strategy"],
        "assessment_context_k": ASSESSMENT_CONTEXT_K,
        "current_evidence_bundle": _safe_evidence_bundle(adaptive_record, units_by_id),
    }


def build_initial_assessment_packets(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units = _read_jsonl(repo_root / "data/processed/phase4_5/retrieval_units.jsonl")
    units_by_id = {unit["id"]: unit for unit in units}
    adaptive_records = _read_jsonl(repo_root / "data/processed/phase9a/adaptive_results.jsonl")
    public_records = _read_jsonl(repo_root / "data/processed/phase9a/public_sanity_adaptive_results.jsonl")
    packets = [_assessment_packet(row, units_by_id) for row in adaptive_records]
    public_packets = [_assessment_packet(row, units_by_id) for row in public_records]
    validate_initial_assessment_packets(packets)
    validate_initial_assessment_packets(public_packets)
    return packets, public_packets


def validate_initial_assessment_packets(packets: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    expected_keys = {
        "question_id",
        "original_question",
        "current_strategy",
        "assessment_context_k",
        "current_evidence_bundle",
    }
    for packet in packets:
        qid = packet.get("question_id")
        if set(packet) != expected_keys:
            violations.append(f"{qid}: unsafe packet keys {sorted(packet)}")
        if packet.get("current_strategy") not in STRATEGIES:
            violations.append(f"{qid}: invalid current_strategy")
        if packet.get("assessment_context_k") != ASSESSMENT_CONTEXT_K:
            violations.append(f"{qid}: invalid assessment_context_k")
        bundle = packet.get("current_evidence_bundle")
        if not isinstance(bundle, list) or not 1 <= len(bundle) <= ASSESSMENT_CONTEXT_K:
            violations.append(f"{qid}: invalid evidence bundle size")
            continue
        context_ids = []
        for item in bundle:
            if "context_id" not in item or "retrieval_text" not in item or "kind" not in item:
                violations.append(f"{qid}: incomplete context item")
            forbidden_item_keys = {"score", "rank", "gold", "hard_negative", "answerability", "split"}
            if forbidden_item_keys & set(item):
                violations.append(f"{qid}:{item.get('context_id')}: forbidden context keys")
            context_ids.append(item.get("context_id"))
        if len(context_ids) != len(set(context_ids)):
            violations.append(f"{qid}: duplicate context IDs")
        text = _json_text(packet)
        for term in LEAKAGE_TERMS:
            if term in text:
                violations.append(f"{qid}: leakage term {term}")
    if violations:
        raise ValueError("Unsafe Phase 9B assessment packets: " + "; ".join(violations[:10]))
    return {"passed": True, "packet_count": len(packets)}


def write_initial_assessor_batch_prompt(
    repo_root: Path,
    synthetic_packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
) -> Path:
    prompt_text = (repo_root / ASSESSOR_PROMPT_PATH).read_text(encoding="utf-8").strip()
    lines = [
        prompt_text,
        "",
        "Assess the following runtime-safe evidence bundles. Return JSONL only, preserving input order.",
        "One JSON object per input line. No Markdown fences. No headings. No commentary.",
        "Do not use tools, web, repository files, gold labels, benchmark metadata, oracle data, or outside knowledge.",
        "Do not answer the user question. Assess evidence sufficiency only.",
        "",
        "SYNTHETIC_REQUESTS_JSONL:",
    ]
    lines.extend(json.dumps(packet, ensure_ascii=False, sort_keys=True) for packet in synthetic_packets)
    lines.extend(["", "PUBLIC_SANITY_REQUESTS_JSONL:"])
    lines.extend(json.dumps(packet, ensure_ascii=False, sort_keys=True) for packet in public_packets)
    path = repo_root / INITIAL_BATCH_PROMPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _preserve_raw_file(source: Path, raw_path: Path) -> None:
    if not source.exists():
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        shutil.copy2(source, raw_path)


def _normalize_string_fields(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    if isinstance(value, str):
        after, method = repair_text_encoding(value)
        if method:
            return after, [{"before": value, "after": after, "repair_method": method}]
        return value, changes
    if isinstance(value, list):
        normalized = []
        for index, item in enumerate(value):
            normalized_item, item_changes = _normalize_string_fields(item)
            normalized.append(normalized_item)
            changes.extend({"field": f"[{index}]{change.get('field', '')}", **change} for change in item_changes)
        return normalized, changes
    if isinstance(value, dict):
        normalized_dict = {}
        for key, item in value.items():
            normalized_item, item_changes = _normalize_string_fields(item)
            normalized_dict[key] = normalized_item
            changes.extend({"field": f".{key}{change.get('field', '')}", **change} for change in item_changes)
        return normalized_dict, changes
    return value, changes


def normalize_initial_assessment_inputs(
    repo_root: Path,
    synthetic_packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = [
        (repo_root / INITIAL_ASSESSMENTS_PATH, repo_root / RAW_INITIAL_ASSESSMENTS_PATH, synthetic_packets),
        (repo_root / PUBLIC_INITIAL_ASSESSMENTS_PATH, repo_root / RAW_PUBLIC_INITIAL_ASSESSMENTS_PATH, public_packets),
    ]
    changes: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    normalized_paths: list[Path] = []
    for source_path, raw_path, _packets in paths:
        _preserve_raw_file(source_path, raw_path)
        if not raw_path.exists():
            continue
        records = _read_jsonl(raw_path)
        normalized_records = []
        for record in records:
            normalized, record_changes = _normalize_string_fields(record)
            question_id = normalized.get("question_id", record.get("question_id"))
            for change in record_changes:
                changes.append({"question_id": question_id, **change})
            normalized_records.append(normalized)
        _write_jsonl(source_path, normalized_records)
        raw_paths.append(raw_path)
        normalized_paths.append(source_path)
    report = {
        "phase": "9B",
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


def _packet_context_ids(packet: dict[str, Any]) -> set[str]:
    key = "current_evidence_bundle" if "current_evidence_bundle" in packet else "combined_evidence_pool"
    return {item["context_id"] for item in packet[key]}


def _validate_assessment_record(
    record: dict[str, Any],
    packet: dict[str, Any],
    allowed_overall: set[str],
    initial_stage: bool,
) -> dict[str, Any]:
    expected_stage = "INITIAL" if initial_stage else "FINAL"
    expected_keys = {
        "question_id",
        "assessment_stage",
        "evidence_requirements",
        "overall_status",
        "missing_evidence_summary",
        "selected_context_ids",
        "correction_strategy",
    }
    if set(record) != expected_keys:
        raise EvidenceAssessmentImportError(
            f"{record.get('question_id')} invalid assessment keys: {sorted(record)}"
        )
    question_id = record["question_id"]
    if question_id != packet["question_id"]:
        raise EvidenceAssessmentImportError(
            f"Assessment order mismatch: expected {packet['question_id']}, got {question_id}"
        )
    if record["assessment_stage"] != expected_stage:
        raise EvidenceAssessmentImportError(
            f"{question_id} invalid assessment_stage: {record['assessment_stage']}"
        )
    valid_context_ids = _packet_context_ids(packet)
    requirements = record["evidence_requirements"]
    if not isinstance(requirements, list) or not requirements:
        raise EvidenceAssessmentImportError(f"{question_id} evidence_requirements must be non-empty")
    requirement_ids: list[str] = []
    for requirement in requirements:
        if set(requirement) != {"requirement_id", "description", "status", "supporting_context_ids"}:
            raise EvidenceAssessmentImportError(
                f"{question_id} invalid requirement keys: {sorted(requirement)}"
            )
        requirement_id = requirement["requirement_id"]
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            raise EvidenceAssessmentImportError(f"{question_id} requirement_id must be non-empty")
        requirement_ids.append(requirement_id)
        if not isinstance(requirement["description"], str) or not requirement["description"].strip():
            raise EvidenceAssessmentImportError(f"{question_id}:{requirement_id} description is empty")
        if requirement["status"] not in EVIDENCE_STATUSES:
            raise EvidenceAssessmentImportError(f"{question_id}:{requirement_id} invalid status")
        supporting = requirement["supporting_context_ids"]
        if not isinstance(supporting, list):
            raise EvidenceAssessmentImportError(
                f"{question_id}:{requirement_id} supporting_context_ids must be a list"
            )
        unknown = sorted(set(supporting) - valid_context_ids)
        if unknown:
            raise EvidenceAssessmentImportError(
                f"{question_id}:{requirement_id} unknown supporting_context_ids: {unknown}"
            )
    if len(requirement_ids) != len(set(requirement_ids)):
        raise EvidenceAssessmentImportError(f"{question_id} duplicate requirement IDs")
    overall_status = record["overall_status"]
    if overall_status not in allowed_overall:
        raise EvidenceAssessmentImportError(f"{question_id} invalid overall_status: {overall_status}")
    selected = record["selected_context_ids"]
    if not isinstance(selected, list):
        raise EvidenceAssessmentImportError(f"{question_id} selected_context_ids must be a list")
    if len(selected) > FINAL_CONTEXT_K:
        raise EvidenceAssessmentImportError(f"{question_id} selected_context_ids exceeds {FINAL_CONTEXT_K}")
    if len(selected) != len(set(selected)):
        raise EvidenceAssessmentImportError(f"{question_id} duplicate selected_context_ids")
    unknown_selected = sorted(set(selected) - valid_context_ids)
    if unknown_selected:
        raise EvidenceAssessmentImportError(f"{question_id} unknown selected_context_ids: {unknown_selected}")
    if not isinstance(record["missing_evidence_summary"], str):
        raise EvidenceAssessmentImportError(f"{question_id} missing_evidence_summary must be a string")
    correction_strategy = record["correction_strategy"]
    if correction_strategy not in CORRECTION_STRATEGIES:
        raise EvidenceAssessmentImportError(f"{question_id} invalid correction_strategy: {correction_strategy}")
    if initial_stage:
        current_strategy = packet["current_strategy"]
        if overall_status == "SUFFICIENT" and correction_strategy != "NONE":
            raise EvidenceAssessmentImportError(f"{question_id} SUFFICIENT requires correction_strategy NONE")
        if overall_status == "NEEDS_CORRECTION":
            if correction_strategy == "NONE":
                raise EvidenceAssessmentImportError(f"{question_id} NEEDS_CORRECTION requires a strategy")
            if correction_strategy == current_strategy:
                raise EvidenceAssessmentImportError(
                    f"{question_id} correction_strategy must differ from current strategy"
                )
    else:
        if overall_status == "NEEDS_CORRECTION":
            raise EvidenceAssessmentImportError(f"{question_id} FINAL cannot use NEEDS_CORRECTION")
        if correction_strategy != "NONE":
            raise EvidenceAssessmentImportError(f"{question_id} final assessment correction_strategy must be NONE")
    text = _json_text(record)
    for term in LEAKAGE_TERMS:
        if term in text:
            raise EvidenceAssessmentImportError(f"{question_id} contains leakage term: {term}")
    return record


def validate_initial_assessments(
    assessment_path: Path, packets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not assessment_path.exists():
        raise FileNotFoundError(f"Missing external initial assessment file: {assessment_path}")
    records = _read_jsonl(assessment_path)
    expected_ids = [packet["question_id"] for packet in packets]
    actual_ids = [record.get("question_id") for record in records]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise EvidenceAssessmentImportError(
            f"Assessment IDs/order mismatch. missing={missing}; unexpected={unexpected}"
        )
    if len(actual_ids) != len(set(actual_ids)):
        raise EvidenceAssessmentImportError("Duplicate assessment question IDs")
    return [
        _validate_assessment_record(record, packet, INITIAL_OVERALL_STATUSES, initial_stage=True)
        for record, packet in zip(records, packets, strict=True)
    ]


def validate_post_correction_assessments(
    assessment_path: Path, packets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not assessment_path.exists():
        raise FileNotFoundError(f"Missing external post-correction assessment file: {assessment_path}")
    records = _read_jsonl(assessment_path)
    expected_ids = [packet["question_id"] for packet in packets]
    actual_ids = [record.get("question_id") for record in records]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise EvidenceAssessmentImportError(
            f"Post-correction IDs/order mismatch. missing={missing}; unexpected={unexpected}"
        )
    return [
        _validate_assessment_record(record, packet, FINAL_OVERALL_STATUSES, initial_stage=False)
        for record, packet in zip(records, packets, strict=True)
    ]


def assessment_to_ledger(record: dict[str, Any], stage: str) -> EvidenceLedger:
    requirements = tuple(
        EvidenceRequirement(
            requirement_id=req["requirement_id"],
            description=req["description"],
            status=req["status"],
            supporting_context_ids=tuple(req["supporting_context_ids"]),
        )
        for req in record["evidence_requirements"]
    )
    return EvidenceLedger(
        question_id=record["question_id"],
        assessment_stage=stage,
        evidence_requirements=requirements,
        overall_status=record["overall_status"],
        missing_evidence_summary=record["missing_evidence_summary"],
        selected_context_ids=tuple(record["selected_context_ids"]),
        correction_strategy=record["correction_strategy"],
    )


def build_combined_evidence_pool(
    initial_record: dict[str, Any],
    corrective_record: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine initial Top-10 and corrective Top-10 without gold-based filtering."""

    def rank_value(result: dict[str, Any]) -> int:
        for key in ["adaptive_rank", "hybrid_rank", "multi_query_rank", "decomposition_rank"]:
            value = result.get(key)
            if isinstance(value, int):
                return value
        raise KeyError(f"No supported rank key found in result for {result.get('retrieval_unit_id')}")

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for branch, row in [("initial", initial_record), ("corrective", corrective_record)]:
        for result in sorted(row["results"], key=rank_value)[:ASSESSMENT_CONTEXT_K]:
            unit_id = result["retrieval_unit_id"]
            if unit_id in seen:
                for existing in pool:
                    if existing["context_id"] == unit_id:
                        existing["branch_provenance"].append(branch)
                continue
            item = _context_from_unit(units_by_id[unit_id])
            item["branch_provenance"] = [branch]
            pool.append(item)
            seen.add(unit_id)
    return pool


def build_post_correction_packet(
    question_id: str,
    original_question: str,
    combined_evidence_pool: list[dict[str, Any]],
    initial_ledger: EvidenceLedger,
) -> dict[str, Any]:
    return {
        "assessment_stage": "FINAL",
        "question_id": question_id,
        "original_question": original_question,
        "combined_evidence_pool": combined_evidence_pool,
        "previous_evidence_requirements": [asdict(req) for req in initial_ledger.evidence_requirements],
        "previous_missing_evidence_summary": initial_ledger.missing_evidence_summary,
    }


def make_trace_event(
    event_type: TraceEventType,
    actor: str,
    input_ids: list[str] | None = None,
    output_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if event_type not in TRACE_EVENT_TYPES:
        raise ValueError(f"Invalid trace event type: {event_type}")
    return _trace_event(event_type, actor, input_ids=input_ids, output_ids=output_ids, metadata=metadata)


def validate_runtime_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = ["gold", "hard_negative", "answerability", "benchmark", "split", "oracle", "metric"]
    violations = []
    for record in records:
        text = _json_text(record.get("trace", []))
        hits = [term for term in forbidden if term in text]
        if hits:
            violations.append({"question_id": record.get("question_id"), "terms": hits})
    return {"passed": not violations, "violations": violations}


def _selected_generation_packet(
    question_id: str,
    query: str,
    selected_context_ids: list[str],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> dict[str, Any]:
    context_items = [_context_item(units_by_id[context_id]) for context_id in selected_context_ids]
    generator_input = format_generator_input(query, context_items)
    return {
        "question_id": question_id,
        "query": query,
        "retrieval_unit_ids": selected_context_ids,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def validate_generator_packets_safe(
    packets: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    validation = validate_packets(packets, units_by_id)
    forbidden = [
        "initial strategy",
        "correction_strategy",
        "evidence ledger",
        "overall_status",
        "retrieval rank",
        "retrieval score",
        "oracle",
        "hard_negative",
        "gold",
    ]
    hits = []
    for packet in packets:
        text = packet["generator_input"].lower()
        for term in forbidden:
            if term in text:
                hits.append({"question_id": packet["question_id"], "term": term})
    return {
        "passed": validation["passed"] and not hits,
        "base_validation": validation,
        "phase9b_forbidden_hits": hits,
    }


def _manifest_hashes(paths: list[Path], repo_root: Path) -> dict[str, str]:
    return {_relative(path, repo_root): sha256_for_file(path) for path in paths if path.exists()}


def _retrieval_tool_manifest() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "name": tool.name,
            "source_phase": tool.source_phase,
            "result_path": tool.result_path.as_posix(),
            "rank_key": tool.rank_key,
            "description": tool.description,
        }
        for name, tool in retrieval_tool_registry().items()
    }


def _write_waiting_manifest(
    repo_root: Path,
    outputs: list[Path],
    missing: list[Path],
    packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / MANIFEST_PATH
    manifest = {
        "phase": "9B",
        "version": "1.0.0",
        "status": "waiting_for_external_initial_assessments",
        "objective": "Bounded corrective retrieval with explicit Evidence Ledger",
        "assessment_context_k": ASSESSMENT_CONTEXT_K,
        "one_corrective_cycle_maximum": True,
        "new_retrieval_algorithms_added": False,
        "answer_generator_called": False,
        "phase9a_router_decisions_modified": False,
        "packet_counts": {"synthetic": len(packets), "public_sanity": len(public_packets)},
        "missing_assessment_files": [_relative(path, repo_root) for path in missing],
        "retrieval_tool_registry": _retrieval_tool_manifest(),
        "assessor_prompt": {
            "version": ASSESSOR_PROMPT_VERSION,
            "path": ASSESSOR_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / ASSESSOR_PROMPT_PATH),
        },
        "output_artifact_hashes": _manifest_hashes(outputs, repo_root),
        "leakage_validation": {
            "initial_packets": validate_initial_assessment_packets(packets),
            "public_initial_packets": validate_initial_assessment_packets(public_packets),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def _build_post_packets_from_initial(
    assessments: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    adaptive_records: dict[str, dict[str, Any]],
    strategy_records: dict[RouterStrategy, dict[str, dict[str, Any]]],
    units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    post_packets: list[dict[str, Any]] = []
    for assessment, packet in zip(assessments, packets, strict=True):
        if assessment["overall_status"] == "SUFFICIENT":
            continue
        qid = assessment["question_id"]
        corrective_strategy = assessment["correction_strategy"]
        corrective_row = strategy_records[corrective_strategy][qid]
        initial_row = adaptive_records[qid]
        pool = build_combined_evidence_pool(initial_row, corrective_row, units_by_id)
        ledger = assessment_to_ledger(assessment, "initial")
        post_packets.append(build_post_correction_packet(qid, packet["original_question"], pool, ledger))
    return post_packets


def write_post_correction_assessor_batch_prompt(
    repo_root: Path,
    synthetic_packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
) -> Path:
    prompt_text = (repo_root / FINAL_ASSESSOR_PROMPT_PATH).read_text(encoding="utf-8").strip()
    lines = [
        prompt_text,
        "",
        "This is the final post-correction assessment. No additional correction is allowed.",
        "Return JSONL only. overall_status must be SUFFICIENT or INSUFFICIENT_WITH_RESIDUAL.",
        "correction_strategy must be NONE.",
        "",
        "SYNTHETIC_POST_CORRECTION_REQUESTS_JSONL:",
    ]
    lines.extend(json.dumps(packet, ensure_ascii=False, sort_keys=True) for packet in synthetic_packets)
    lines.extend(["", "PUBLIC_SANITY_POST_CORRECTION_REQUESTS_JSONL:"])
    lines.extend(json.dumps(packet, ensure_ascii=False, sort_keys=True) for packet in public_packets)
    path = repo_root / POST_BATCH_PROMPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_post_correction_single_prompts(
    repo_root: Path,
    synthetic_packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
) -> Path:
    output_dir = repo_root / FINAL_SINGLE_PROMPT_DIR
    raw_output_dir = output_dir / "raw_outputs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = (repo_root / FINAL_ASSESSOR_PROMPT_PATH).read_text(encoding="utf-8").strip()
    records = [*synthetic_packets, *public_packets]
    manifest_records = []
    for index, packet in enumerate(records, start=1):
        qid = packet["question_id"]
        allowed_context_ids = [item["context_id"] for item in packet["combined_evidence_pool"]]
        lines = [
            prompt_text,
            "",
            "Assess exactly ONE runtime-safe Phase 9B FINAL post-correction record.",
            "",
            "ASSESSMENT_STAGE = FINAL",
            "",
            "ALLOWED_CONTEXT_IDS:",
        ]
        lines.extend(f"- {context_id}" for context_id in allowed_context_ids)
        lines.extend(
            [
                "",
                (
                    "supporting_context_ids and selected_context_ids may contain ONLY IDs copied verbatim "
                    "from ALLOWED_CONTEXT_IDS. If no supplied evidence supports a requirement, use the "
                    "appropriate incomplete status and an empty supporting ID list. Never construct or infer "
                    "a context ID."
                ),
                "",
                "Return ONLY valid JSONL.",
                "",
                "Requirements:",
                "- exactly one JSON object",
                "- one JSON object per line",
                "- assessment_stage must be FINAL",
                "- overall_status must be SUFFICIENT or INSUFFICIENT_WITH_RESIDUAL",
                "- correction_strategy must be NONE",
                "- no Markdown code fences",
                "- no headings",
                "- no commentary outside the JSON object",
                "- do not use tools, files, web search, repository access, or outside information",
                "- use only the supplied QUESTION and COMBINED EVIDENCE",
                "- do not answer the user question",
                "- assess final evidence sufficiency only",
                "",
                "REQUEST_JSON:",
                json.dumps(packet, ensure_ascii=False, sort_keys=True),
            ]
        )
        prompt = "\n".join(lines) + "\n"
        prompt_path = output_dir / f"{qid}.txt"
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
        manifest_records.append(
            {
                "order": index,
                "question_id": qid,
                "prompt_path": _relative(prompt_path, repo_root),
                "prompt_sha256": _hash_text(prompt),
                "character_count": len(prompt),
                "supplied_context_ids": allowed_context_ids,
                "context_count": len(allowed_context_ids),
                "model": "gpt-5.5",
                "reasoning_effort": "low",
            }
        )

    manifest = {
        "phase": "9B",
        "protocol": "final_v1_single_question_fresh_assessor_process",
        "total_synthetic_questions": len(synthetic_packets),
        "total_public_sanity_questions": len(public_packets),
        "total_questions": len(records),
        "model": "gpt-5.5",
        "reasoning_effort": "low",
        "final_assessor_prompt": {
            "version": FINAL_ASSESSOR_PROMPT_VERSION,
            "path": FINAL_ASSESSOR_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / FINAL_ASSESSOR_PROMPT_PATH),
        },
        "records": manifest_records,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path



def normalize_post_correction_assessment_inputs(
    repo_root: Path,
    synthetic_packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = [
        (repo_root / POST_ASSESSMENTS_PATH, repo_root / RAW_POST_ASSESSMENTS_PATH, synthetic_packets),
        (repo_root / PUBLIC_POST_ASSESSMENTS_PATH, repo_root / RAW_PUBLIC_POST_ASSESSMENTS_PATH, public_packets),
    ]
    changes: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    normalized_paths: list[Path] = []
    for source_path, raw_path, _packets in paths:
        _preserve_raw_file(source_path, raw_path)
        if not raw_path.exists():
            continue
        records = _read_jsonl(raw_path)
        normalized_records = []
        for record in records:
            normalized, record_changes = _normalize_string_fields(record)
            question_id = normalized.get("question_id", record.get("question_id"))
            for change in record_changes:
                changes.append({"question_id": question_id, **change})
            normalized_records.append(normalized)
        _write_jsonl(source_path, normalized_records)
        raw_paths.append(raw_path)
        normalized_paths.append(source_path)
    report = {
        "phase": "9B",
        "stage": "post_correction_final",
        "normalization": "deterministic transport encoding repair only",
        "raw_file_hashes": {_relative(path, repo_root): sha256_for_file(path) for path in raw_paths},
        "normalized_file_hashes": {
            _relative(path, repo_root): sha256_for_file(path) for path in normalized_paths
        },
        "change_count": len(changes),
        "changes": changes,
    }
    _write_json(repo_root / (PHASE9B_ROOT / "post_correction_encoding_normalization_report.json"), report)
    return report


def _result_rank_value(result: dict[str, Any]) -> int:
    for key in [
        "adaptive_rank",
        "hybrid_rank",
        "multi_query_rank",
        "decomposition_rank",
        "rank",
    ]:
        value = result.get(key)
        if isinstance(value, int):
            return value
    raise KeyError(f"No supported rank key found in result for {result.get('retrieval_unit_id')}")


def _result_score_value(result: dict[str, Any]) -> float:
    for key in [
        "adaptive_score",
        "multi_query_rrf_score",
        "decomposition_rrf_score",
        "rrf_score",
        "score",
    ]:
        value = result.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _ordered_result_ids(row: dict[str, Any], limit: int | None = None) -> list[str]:
    results = sorted(row["results"], key=_result_rank_value)
    if limit is not None:
        results = results[:limit]
    return [result["retrieval_unit_id"] for result in results]


def _ranked_for_selected_metrics(selected_by_id: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    return {
        qid: [
            {"retrieval_unit_id": unit_id, "score": float(len(ids) - index)}
            for index, unit_id in enumerate(ids)
        ]
        for qid, ids in selected_by_id.items()
    }


def _selected_metrics_view(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = ["hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5", "mrr", "ndcg@10"]
    return {
        "definitions": metrics["definitions"],
        "overall_non_abstain": {
            key: metrics["overall_non_abstain"].get(key) for key in keep if key in metrics["overall_non_abstain"]
        },
        "by_split": {
            split: {key: values.get(key) for key in keep if key in values}
            for split, values in metrics["by_split"].items()
        },
        "per_question": metrics["per_question"],
        "hard_negative_analysis": metrics["hard_negative_analysis"],
    }


def _candidate_discovery_metrics(
    questions: list[dict[str, Any]],
    candidate_ids_by_question: dict[str, list[str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, float]]] = {}
    overall_rows: list[dict[str, float]] = []
    for question in questions:
        qid = question["question_id"]
        gold = set(question.get("gold_chunk_ids", []))
        candidates = candidate_ids_by_question[qid]
        metrics: dict[str, float] = {}
        for k in [10, 20, 50]:
            top_k = candidates[:k]
            metrics[f"gold_recall@{k}"] = len(set(top_k) & gold) / len(gold) if gold else 0.0
        rows.append(
            {
                "question_id": qid,
                "split": question.get("split"),
                "answerability": question.get("answerability"),
                "gold_chunk_count": len(gold),
                "candidate_count": len(candidates),
                "metrics": metrics,
            }
        )
        if question.get("answerability") != "abstain":
            overall_rows.append(metrics)
            by_split.setdefault(str(question.get("split")), []).append(metrics)

    def aggregate(values: list[dict[str, float]]) -> dict[str, float]:
        if not values:
            return {}
        keys = sorted({key for value in values for key in value})
        return {key: sum(value[key] for value in values) / len(values) for key in keys}

    return {
        "definition": "Gold recall over the runtime candidate pool before final evidence selection; non-abstain denominator for aggregates.",
        "overall_non_abstain": aggregate(overall_rows),
        "by_split": {split: aggregate(values) for split, values in sorted(by_split.items())},
        "per_question": rows,
    }


def _sufficiency_answerability_diagnostics(
    questions: list[dict[str, Any]], final_status_by_id: dict[str, str]
) -> dict[str, Any]:
    counts = {
        "true_sufficient": 0,
        "false_sufficient": 0,
        "true_insufficient": 0,
        "false_insufficient": 0,
    }
    rows = []
    for question in questions:
        qid = question["question_id"]
        answerable = question.get("answerability") in {"answer", "correct_premise"}
        sufficient = final_status_by_id[qid] == "SUFFICIENT"
        if answerable and sufficient:
            bucket = "true_sufficient"
        elif not answerable and sufficient:
            bucket = "false_sufficient"
        elif not answerable and not sufficient:
            bucket = "true_insufficient"
        else:
            bucket = "false_insufficient"
        counts[bucket] += 1
        rows.append(
            {
                "question_id": qid,
                "benchmark_answerable": answerable,
                "final_runtime_sufficient": sufficient,
                "diagnostic_bucket": bucket,
            }
        )
    precision_den = counts["true_sufficient"] + counts["false_sufficient"]
    recall_den = counts["true_sufficient"] + counts["false_insufficient"]
    residual_den = counts["true_insufficient"] + counts["false_sufficient"]
    return {
        "label": "OFFLINE benchmark evaluation only; not used at runtime.",
        "counts": counts,
        "sufficiency_precision": counts["true_sufficient"] / precision_den if precision_den else None,
        "sufficiency_recall": counts["true_sufficient"] / recall_den if recall_den else None,
        "residual_detection_precision": counts["true_insufficient"] / residual_den if residual_den else None,
        "per_question": rows,
    }


def _hard_negative_selection_diagnostics(
    questions: list[dict[str, Any]], selected_by_id: dict[str, list[str]]) -> dict[str, Any]:
    rows = []
    for question in questions:
        qid = question["question_id"]
        selected = selected_by_id[qid]
        gold = set(question.get("gold_chunk_ids", []))
        hard = set(question.get("hard_negative_chunk_ids", []))
        first_gold = next((index + 1 for index, unit_id in enumerate(selected) if unit_id in gold), None)
        first_hard = next((index + 1 for index, unit_id in enumerate(selected) if unit_id in hard), None)
        hard_selected = [unit_id for unit_id in selected if unit_id in hard]
        rows.append(
            {
                "question_id": qid,
                "selected_hard_negative_ids": hard_selected,
                "first_gold_selected_rank": first_gold,
                "first_hard_negative_selected_rank": first_hard,
                "hard_negative_selected_above_first_gold": first_hard is not None
                and (first_gold is None or first_hard < first_gold),
            }
        )
    synq_012_selected = selected_by_id.get("SYNQ-012-A", [])
    return {
        "selected_hard_negative_count": sum(bool(row["selected_hard_negative_ids"]) for row in rows),
        "hard_negative_above_first_gold_count": sum(
            row["hard_negative_selected_above_first_gold"] for row in rows
        ),
        "superseded_rca_selected_as_current_evidence_SYNQ_012_A": any(
            "RCA-B017-004" in unit_id and "RCA-B017-004-rev2" not in unit_id
            for unit_id in synq_012_selected
        ),
        "current_revision_evidence_retained_SYNQ_012_A": any(
            "RCA-B017-004-rev2" in unit_id for unit_id in synq_012_selected
        ),
        "per_question": rows,
    }


def _dedupe_ordered(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in ids:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _retrieval_query_count(strategy: str, row: dict[str, Any]) -> int:
    if strategy == "HYBRID":
        return 1
    if strategy == "MULTI_QUERY":
        return 4
    if strategy == "DECOMPOSITION":
        return int(row.get("retrieval_query_count", 1) or 1)
    return 0


def _make_phase9b_trace(
    qid: str,
    initial_strategy: str,
    initial_assessment: dict[str, Any],
    initial_candidate_ids: list[str],
    correction_performed: bool,
    corrective_strategy: str,
    corrective_candidate_ids: list[str],
    final_assessment: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    trace = [
        make_trace_event(
            "QUESTION_RECEIVED",
            "phase9b_controller",
            input_ids=[qid],
            output_ids=[f"question:{qid}"],
            metadata={"question_id": qid},
        ),
        make_trace_event(
            "ROUTE_SELECTED",
            "phase9a_router",
            input_ids=[f"question:{qid}"],
            output_ids=[f"strategy:{initial_strategy}"],
            metadata={"selected_strategy": initial_strategy},
        ),
        make_trace_event(
            "RETRIEVAL_COMPLETED",
            "retrieval_tool_registry",
            input_ids=[f"strategy:{initial_strategy}"],
            output_ids=initial_candidate_ids,
            metadata={"strategy": initial_strategy, "context_count": len(initial_candidate_ids)},
        ),
        make_trace_event(
            "EVIDENCE_ASSESSED",
            "isolated_evidence_assessor_initial_v2",
            input_ids=initial_candidate_ids,
            output_ids=[f"ledger:{qid}:initial"],
            metadata={
                "overall_status": initial_assessment["overall_status"],
                "selected_context_ids": initial_assessment["selected_context_ids"],
                "requirement_statuses": [
                    req["status"] for req in initial_assessment["evidence_requirements"]
                ],
            },
        ),
    ]
    if correction_performed:
        trace.extend(
            [
                make_trace_event(
                    "CORRECTION_SELECTED",
                    "isolated_evidence_assessor_initial_v2",
                    input_ids=[f"ledger:{qid}:initial"],
                    output_ids=[f"strategy:{corrective_strategy}"],
                    metadata={"correction_strategy": corrective_strategy},
                ),
                make_trace_event(
                    "CORRECTIVE_RETRIEVAL_COMPLETED",
                    "retrieval_tool_registry",
                    input_ids=[f"strategy:{corrective_strategy}"],
                    output_ids=corrective_candidate_ids,
                    metadata={
                        "strategy": corrective_strategy,
                        "context_count": len(corrective_candidate_ids),
                    },
                ),
            ]
        )
        if final_assessment is not None:
            trace.append(
                make_trace_event(
                    "FINAL_EVIDENCE_ASSESSED",
                    "isolated_evidence_assessor_final_v1",
                    input_ids=_dedupe_ordered([*initial_candidate_ids, *corrective_candidate_ids]),
                    output_ids=[f"ledger:{qid}:final"],
                    metadata={
                        "overall_status": final_assessment["overall_status"],
                        "selected_context_ids": final_assessment["selected_context_ids"],
                        "requirement_statuses": [
                            req["status"] for req in final_assessment["evidence_requirements"]
                        ],
                    },
                )
            )
    return trace


def _build_case_studies(
    case_ids: list[str],
    questions_by_id: dict[str, dict[str, Any]],
    initial_assessments_by_id: dict[str, dict[str, Any]],
    final_assessments_by_id: dict[str, dict[str, Any]],
    corrective_results_by_id: dict[str, dict[str, Any]],
    public_reference_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for qid in case_ids:
        if qid not in corrective_results_by_id:
            continue
        runtime = corrective_results_by_id[qid]
        selected = runtime["final_selected_context_ids"]
        row: dict[str, Any] = {
            "runtime": {
                "initial_strategy": runtime["initial_strategy"],
                "initial_evidence_status": runtime["initial_overall_status"],
                "correction_occurred": runtime["correction_performed"],
                "corrective_strategy": runtime["correction_strategy"],
                "final_evidence_status": runtime["final_overall_status"],
                "final_selected_context_ids": selected,
            }
        }
        if qid in questions_by_id:
            question = questions_by_id[qid]
            gold = set(question.get("gold_chunk_ids", []))
            hard = set(question.get("hard_negative_chunk_ids", []))
            row["offline_evaluation"] = {
                "gold_chunk_ids": list(question.get("gold_chunk_ids", [])),
                "selected_gold_ids": [unit_id for unit_id in selected if unit_id in gold],
                "selected_hard_negative_ids": [unit_id for unit_id in selected if unit_id in hard],
            }
        elif qid in public_reference_by_id:
            expected = set(public_reference_by_id[qid].get("expected_retrieval_unit_ids", []))
            row["offline_public_reference_check"] = {
                "expected_retrieval_unit_ids": list(expected),
                "selected_expected_ids": [unit_id for unit_id in selected if unit_id in expected],
            }
        if qid in final_assessments_by_id:
            row["runtime"]["final_assessor_record_present"] = True
        row["runtime"]["initial_selected_context_ids"] = initial_assessments_by_id[qid]["selected_context_ids"]
        rows[qid] = row
    return rows


def _build_phase9b_records(
    assessments: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    post_assessments_by_id: dict[str, dict[str, Any]],
    adaptive_records: dict[str, dict[str, Any]],
    strategy_records: dict[RouterStrategy, dict[str, dict[str, Any]]],
    units_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
    result_records: list[dict[str, Any]] = []
    ledger_records: list[dict[str, Any]] = []
    selected_by_id: dict[str, list[str]] = {}
    candidate_pool_by_id: dict[str, list[str]] = {}
    packet_by_id = {packet["question_id"]: packet for packet in packets}
    for initial in assessments:
        qid = initial["question_id"]
        packet = packet_by_id[qid]
        adaptive_row = adaptive_records[qid]
        initial_strategy = adaptive_row["selected_strategy"]
        initial_candidate_ids = _ordered_result_ids(adaptive_row, ASSESSMENT_CONTEXT_K)
        correction_performed = initial["overall_status"] == "NEEDS_CORRECTION"
        corrective_strategy = initial["correction_strategy"] if correction_performed else "NONE"
        corrective_candidate_ids: list[str] = []
        final_assessment: dict[str, Any] | None = None
        initial_query_count = int(
            adaptive_row.get("strategy_provenance", {}).get(
                "retrieval_query_count", _retrieval_query_count(initial_strategy, adaptive_row)
            )
        )
        corrective_query_count = 0
        combined_pool_ids = list(initial_candidate_ids)
        if correction_performed:
            corrective_row = strategy_records[corrective_strategy][qid]
            corrective_query_count = _retrieval_query_count(corrective_strategy, corrective_row)
            corrective_candidate_ids = _ordered_result_ids(corrective_row, ASSESSMENT_CONTEXT_K)
            combined_pool_ids = _dedupe_ordered([*initial_candidate_ids, *corrective_candidate_ids])
            final_assessment = post_assessments_by_id[qid]
            final_status = final_assessment["overall_status"]
            selected_ids = final_assessment["selected_context_ids"]
            final_ledger_record = final_assessment
        else:
            final_status = "SUFFICIENT"
            selected_ids = initial["selected_context_ids"]
            final_ledger_record = {
                **initial,
                "assessment_stage": "FINAL",
                "overall_status": "SUFFICIENT",
                "correction_strategy": "NONE",
            }
        selected_by_id[qid] = list(selected_ids)
        candidate_pool_by_id[qid] = combined_pool_ids
        trace = _make_phase9b_trace(
            qid,
            initial_strategy,
            initial,
            initial_candidate_ids,
            correction_performed,
            corrective_strategy,
            corrective_candidate_ids,
            final_assessment,
        )
        result_record = {
            "question_id": qid,
            "query": packet["original_question"],
            "initial_strategy": initial_strategy,
            "initial_overall_status": initial["overall_status"],
            "correction_performed": correction_performed,
            "correction_strategy": corrective_strategy,
            "final_overall_status": final_status,
            "initial_candidate_ids": initial_candidate_ids,
            "corrective_candidate_ids": corrective_candidate_ids,
            "combined_evidence_pool_ids": combined_pool_ids,
            "initial_retrieval_query_count": initial_query_count,
            "corrective_retrieval_query_count": corrective_query_count,
            "total_retrieval_query_count": initial_query_count + corrective_query_count,
            "retrieval_strategy_execution_count": 2 if correction_performed else 1,
            "initial_selected_context_ids": initial["selected_context_ids"],
            "final_selected_context_ids": list(selected_ids),
            "trace": trace,
        }
        result_records.append(result_record)
        ledger_records.append(
            {
                "question_id": qid,
                "assessment_stage": "FINAL",
                "evidence_requirements": final_ledger_record["evidence_requirements"],
                "overall_status": final_status,
                "missing_evidence_summary": final_ledger_record["missing_evidence_summary"],
                "selected_context_ids": list(selected_ids),
                "correction_strategy": "NONE",
            }
        )
    if not validate_runtime_trace(result_records)["passed"]:
        raise ValueError("Phase 9B runtime trace contains forbidden evaluation metadata")
    for selected in selected_by_id.values():
        for context_id in selected:
            if context_id not in units_by_id:
                raise ValueError(f"Unknown selected context ID: {context_id}")
    return result_records, ledger_records, selected_by_id, candidate_pool_by_id


def _runtime_behavior_diagnostics(
    initial_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    result_records: list[dict[str, Any]],
) -> dict[str, Any]:
    question_count = len(initial_records)
    needs = [row for row in initial_records if row["overall_status"] == "NEEDS_CORRECTION"]
    sufficient = [row for row in initial_records if row["overall_status"] == "SUFFICIENT"]
    final_by_id = {row["question_id"]: row for row in final_records}
    resolved = sum(final_by_id[row["question_id"]]["overall_status"] == "SUFFICIENT" for row in needs)
    unresolved = len(needs) - resolved
    strategy_counts = {strategy: 0 for strategy in STRATEGIES}
    for row in needs:
        strategy_counts[row["correction_strategy"]] += 1
    final_selected_counts = []
    for row in initial_records:
        if row["overall_status"] == "SUFFICIENT":
            final_selected_counts.append(len(row["selected_context_ids"]))
        else:
            final_selected_counts.append(len(final_by_id[row["question_id"]]["selected_context_ids"]))
    return {
        "initially_sufficient": len(sufficient),
        "initially_needs_correction": len(needs),
        "correction_rate": len(needs) / question_count if question_count else 0.0,
        "correction_strategy_counts": strategy_counts,
        "hybrid_corrections": strategy_counts["HYBRID"],
        "multi_query_corrections": strategy_counts["MULTI_QUERY"],
        "decomposition_corrections": strategy_counts["DECOMPOSITION"],
        "resolved_after_correction": resolved,
        "still_insufficient_with_residual": unresolved,
        "mean_retrieval_strategies_per_question": sum(
            row["retrieval_strategy_execution_count"] for row in result_records
        ) / question_count
        if question_count
        else 0.0,
        "total_retrieval_queries_implied": sum(row["total_retrieval_query_count"] for row in result_records),
        "mean_final_evidence_items_selected": sum(final_selected_counts) / len(final_selected_counts)
        if final_selected_counts
        else 0.0,
    }


def _comparison_table(phase9a_metrics: dict[str, Any], selected_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = ["hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5", "mrr", "ndcg@10"]
    base = phase9a_metrics["adaptive_router"]["overall_non_abstain"]
    selected = selected_metrics["overall_non_abstain"]
    return [
        {
            "metric": metric,
            "phase9a_adaptive_initial": base.get(metric),
            "phase9b_final_selected_evidence": selected.get(metric),
            "delta": None if metric not in base or metric not in selected else selected[metric] - base[metric],
        }
        for metric in ordered
    ]


def _write_complete_manifest(
    repo_root: Path,
    outputs: list[Path],
    metrics_payload: dict[str, Any],
    final_run_report: dict[str, Any] | None,
    initial_normalization_report: dict[str, Any],
    final_normalization_report: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / MANIFEST_PATH
    manifest = {
        "phase": "9B",
        "version": "1.0.0",
        "status": "complete",
        "objective": "Bounded corrective retrieval with explicit Evidence Ledger",
        "one_corrective_cycle_maximum": True,
        "new_retrieval_algorithms_added": False,
        "answer_generator_called": False,
        "phase10_started": False,
        "phase9a_router_decisions_modified": False,
        "assessment_context_k": ASSESSMENT_CONTEXT_K,
        "final_context_k": FINAL_CONTEXT_K,
        "initial_assessor": {
            "version": ASSESSOR_PROMPT_VERSION,
            "path": ASSESSOR_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / ASSESSOR_PROMPT_PATH),
        },
        "final_assessor": {
            "version": FINAL_ASSESSOR_PROMPT_VERSION,
            "path": FINAL_ASSESSOR_PROMPT_PATH.as_posix(),
            "sha256": sha256_for_file(repo_root / FINAL_ASSESSOR_PROMPT_PATH),
            "run_report": final_run_report,
        },
        "retrieval_tool_registry": _retrieval_tool_manifest(),
        "runtime_behavior": metrics_payload["runtime_evidence_behavior"],
        "selected_evidence_metrics": metrics_payload["selected_evidence_metrics"]["overall_non_abstain"],
        "candidate_discovery": metrics_payload["candidate_discovery"],
        "sufficiency_answerability": metrics_payload["sufficiency_answerability_evaluation"]["counts"],
        "initial_encoding_normalization": initial_normalization_report,
        "final_encoding_normalization": final_normalization_report,
        "output_artifact_hashes": _manifest_hashes(outputs, repo_root),
    }
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def _write_post_waiting_manifest(
    repo_root: Path,
    outputs: list[Path],
    missing: list[Path],
    packets: list[dict[str, Any]],
    public_packets: list[dict[str, Any]],
    post_packets: list[dict[str, Any]],
    public_post_packets: list[dict[str, Any]],
    normalization_report: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / MANIFEST_PATH
    manifest = {
        "phase": "9B",
        "version": "1.0.0",
        "status": "waiting_for_external_post_correction_assessments",
        "initial_assessment_files_validated": True,
        "post_correction_packet_counts": {"synthetic": len(post_packets), "public_sanity": len(public_post_packets)},
        "missing_assessment_files": [_relative(path, repo_root) for path in missing],
        "encoding_normalization": normalization_report,
        "packet_counts": {"synthetic": len(packets), "public_sanity": len(public_packets)},
        "one_corrective_cycle_maximum": True,
        "output_artifact_hashes": _manifest_hashes(outputs, repo_root),
    }
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def run_phase9b(repo_root: Path | None = None) -> Phase9BResult:
    """Run Phase 9B until the next required external Evidence Assessor gate."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / PHASE9B_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    phase9a_result = run_phase9a(repo_root)
    if phase9a_result.status != "complete":
        raise RuntimeError(f"Phase 9A must be complete before Phase 9B, got {phase9a_result.status}")

    packets, public_packets = build_initial_assessment_packets(repo_root)
    packets_path = repo_root / INITIAL_PACKETS_PATH
    public_packets_path = repo_root / PUBLIC_INITIAL_PACKETS_PATH
    _write_jsonl(packets_path, packets)
    _write_jsonl(public_packets_path, public_packets)
    batch_prompt_path = write_initial_assessor_batch_prompt(repo_root, packets, public_packets)
    outputs = [packets_path, public_packets_path, batch_prompt_path]

    required_initial = [repo_root / INITIAL_ASSESSMENTS_PATH, repo_root / PUBLIC_INITIAL_ASSESSMENTS_PATH]
    missing_initial = [path for path in required_initial if not path.exists()]
    if missing_initial:
        manifest, manifest_path = _write_waiting_manifest(
            repo_root, outputs, missing_initial, packets, public_packets
        )
        return Phase9BResult(
            status="waiting_for_external_initial_assessments",
            initial_packets=packets,
            public_initial_packets=public_packets,
            outputs=outputs,
            manifest=manifest,
            manifest_path=manifest_path,
            missing_assessment_files=missing_initial,
        )

    normalization_report = normalize_initial_assessment_inputs(repo_root, packets, public_packets)
    initial_assessments = validate_initial_assessments(repo_root / INITIAL_ASSESSMENTS_PATH, packets)
    public_initial_assessments = validate_initial_assessments(
        repo_root / PUBLIC_INITIAL_ASSESSMENTS_PATH, public_packets
    )

    units = _read_jsonl(repo_root / "data/processed/phase4_5/retrieval_units.jsonl")
    units_by_id = {unit["id"]: unit for unit in units}
    adaptive_records = {
        row["question_id"]: row for row in _read_jsonl(repo_root / "data/processed/phase9a/adaptive_results.jsonl")
    }
    public_adaptive_records = {
        row["question_id"]: row
        for row in _read_jsonl(repo_root / "data/processed/phase9a/public_sanity_adaptive_results.jsonl")
    }
    strategy_records = _load_strategy_records(repo_root, strategy_registry())
    public_strategy_records = _load_strategy_records(repo_root, _public_strategy_registry())

    post_packets = _build_post_packets_from_initial(
        initial_assessments, packets, adaptive_records, strategy_records, units_by_id
    )
    public_post_packets = _build_post_packets_from_initial(
        public_initial_assessments, public_packets, public_adaptive_records, public_strategy_records, units_by_id
    )
    post_packets_path = repo_root / POST_PACKETS_PATH
    public_post_packets_path = repo_root / PUBLIC_POST_PACKETS_PATH
    _write_jsonl(post_packets_path, post_packets)
    _write_jsonl(public_post_packets_path, public_post_packets)
    post_prompt_path = write_post_correction_assessor_batch_prompt(repo_root, post_packets, public_post_packets)
    final_single_manifest_path = repo_root / FINAL_SINGLE_PROMPT_DIR / "manifest.json"
    if not final_single_manifest_path.exists():
        final_single_manifest_path = write_post_correction_single_prompts(
            repo_root, post_packets, public_post_packets
        )
    outputs.extend(
        [
            repo_root / ENCODING_NORMALIZATION_REPORT_PATH,
            post_packets_path,
            public_post_packets_path,
            post_prompt_path,
            final_single_manifest_path,
        ]
    )

    missing_post = [
        path
        for path in [repo_root / POST_ASSESSMENTS_PATH, repo_root / PUBLIC_POST_ASSESSMENTS_PATH]
        if (post_packets or public_post_packets) and not path.exists()
    ]
    if missing_post:
        manifest, manifest_path = _write_post_waiting_manifest(
            repo_root,
            outputs,
            missing_post,
            packets,
            public_packets,
            post_packets,
            public_post_packets,
            normalization_report,
        )
        return Phase9BResult(
            status="waiting_for_external_post_correction_assessments",
            initial_packets=packets,
            public_initial_packets=public_packets,
            outputs=outputs,
            manifest=manifest,
            manifest_path=manifest_path,
            missing_assessment_files=missing_post,
        )

    final_normalization_report = normalize_post_correction_assessment_inputs(
        repo_root, post_packets, public_post_packets
    )
    post_assessments = validate_post_correction_assessments(repo_root / POST_ASSESSMENTS_PATH, post_packets)
    public_post_assessments = validate_post_correction_assessments(
        repo_root / PUBLIC_POST_ASSESSMENTS_PATH, public_post_packets
    )
    post_by_id = {row["question_id"]: row for row in post_assessments}
    public_post_by_id = {row["question_id"]: row for row in public_post_assessments}

    synthetic_results, synthetic_ledgers, selected_by_id, candidates_by_id = _build_phase9b_records(
        initial_assessments,
        packets,
        post_by_id,
        adaptive_records,
        strategy_records,
        units_by_id,
    )
    public_results, public_ledgers, _public_selected_by_id, public_candidates_by_id = _build_phase9b_records(
        public_initial_assessments,
        public_packets,
        public_post_by_id,
        public_adaptive_records,
        public_strategy_records,
        units_by_id,
    )
    all_results = [*synthetic_results, *public_results]
    all_ledgers = [*synthetic_ledgers, *public_ledgers]

    corrective_results_path = output_root / "corrective_results.jsonl"
    public_corrective_results_path = output_root / "public_sanity_corrective_results.jsonl"
    evidence_ledgers_path = output_root / "evidence_ledgers.jsonl"
    generation_packets_path = output_root / "generation_packets.jsonl"
    metrics_path = output_root / "metrics.json"
    public_report_path = output_root / "public_sanity_report.json"

    _write_jsonl(corrective_results_path, synthetic_results)
    _write_jsonl(public_corrective_results_path, public_results)
    _write_jsonl(evidence_ledgers_path, all_ledgers)

    generator_prompt_hash = sha256_for_file(repo_root / GENERATOR_PROMPT_PATH)
    generation_packets = [
        _selected_generation_packet(
            row["question_id"],
            row["query"],
            row["final_selected_context_ids"],
            units_by_id,
            generator_prompt_hash,
        )
        for row in synthetic_results
    ]
    packet_validation = validate_generator_packets_safe(generation_packets, units_by_id)
    if not packet_validation["passed"]:
        raise ValueError(f"Unsafe Phase 9B generation packets: {packet_validation}")
    _write_jsonl(generation_packets_path, generation_packets)

    questions = _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl")
    questions_by_id = {row["question_id"]: row for row in questions}
    selected_metrics = _selected_metrics_view(
        compute_metrics(questions, _ranked_for_selected_metrics(selected_by_id))
    )
    candidate_discovery = _candidate_discovery_metrics(questions, candidates_by_id)
    final_status_by_id = {row["question_id"]: row["final_overall_status"] for row in synthetic_results}
    sufficiency = _sufficiency_answerability_diagnostics(questions, final_status_by_id)
    hard_negative_diagnostics = _hard_negative_selection_diagnostics(questions, selected_by_id)
    phase9a_metrics = json.loads((repo_root / "data/processed/phase9a/metrics.json").read_text(encoding="utf-8"))

    all_initial = [*initial_assessments, *public_initial_assessments]
    all_final = [*post_assessments, *public_post_assessments]
    runtime_behavior = _runtime_behavior_diagnostics(all_initial, all_final, all_results)
    final_status_counts = {
        "SUFFICIENT": sum(row["final_overall_status"] == "SUFFICIENT" for row in all_results),
        "INSUFFICIENT_WITH_RESIDUAL": sum(
            row["final_overall_status"] == "INSUFFICIENT_WITH_RESIDUAL" for row in all_results
        ),
    }

    public_reference_records = _read_jsonl(repo_root / "data/eval/public_sanity/reference.jsonl")
    public_reference_by_id = {row["question_id"]: row for row in public_reference_records}
    public_report_rows = []
    for row in public_results:
        qid = row["question_id"]
        expected = public_reference_by_id.get(qid, {}).get("expected_retrieval_unit_ids", [])
        selected = row["final_selected_context_ids"]
        public_report_rows.append(
            {
                "question_id": qid,
                "initial_strategy": row["initial_strategy"],
                "initial_overall_status": row["initial_overall_status"],
                "correction_performed": row["correction_performed"],
                "correction_strategy": row["correction_strategy"],
                "final_overall_status": row["final_overall_status"],
                "final_selected_context_ids": selected,
                "expected_retrieval_unit_ids": expected,
                "expected_selected_ids": [unit_id for unit_id in selected if unit_id in set(expected)],
                "candidate_pool_expected_ids": [
                    unit_id for unit_id in public_candidates_by_id[qid] if unit_id in set(expected)
                ],
            }
        )
    public_report = {
        "phase": "9B",
        "question_count": len(public_results),
        "report": public_report_rows,
    }
    _write_json(public_report_path, public_report)

    all_results_by_id = {row["question_id"]: row for row in all_results}
    initial_by_id = {row["question_id"]: row for row in all_initial}
    final_by_id = {row["question_id"]: row for row in all_final}
    case_studies = _build_case_studies(
        [
            "SYNQ-001-A",
            "SYNQ-002-A",
            "SYNQ-009-B",
            "SYNQ-012-A",
            "SYNQ-014-B",
            "SYNQ-019-B",
            "PUBSAN-005",
        ],
        questions_by_id,
        initial_by_id,
        final_by_id,
        all_results_by_id,
        public_reference_by_id,
    )

    final_run_report_path = repo_root / FINAL_SINGLE_PROMPT_DIR / "run_report.json"
    final_run_report = (
        json.loads(final_run_report_path.read_text(encoding="utf-8")) if final_run_report_path.exists() else None
    )
    metrics_payload = {
        "phase": "9B",
        "definitions": {
            "runtime_behavior": "Evidence assessor sufficiency and one bounded correction cycle; no gold labels used.",
            "selected_evidence_metrics": "Offline gold-label evaluation over final selected evidence IDs only.",
            "candidate_discovery": "Offline gold recall over initial Top-10 plus corrective Top-10 candidate pools.",
            "sufficiency_answerability": "Offline comparison between runtime sufficiency and benchmark answerability categories.",
        },
        "runtime_evidence_behavior": runtime_behavior,
        "final_status_counts": final_status_counts,
        "selected_evidence_metrics": selected_metrics,
        "candidate_discovery": candidate_discovery,
        "phase9a_initial_adaptive_metrics": phase9a_metrics["adaptive_router"],
        "comparison_table": _comparison_table(phase9a_metrics, selected_metrics),
        "sufficiency_answerability_evaluation": sufficiency,
        "hard_negative_revision_analysis": hard_negative_diagnostics,
        "required_case_studies": case_studies,
        "generator_packet_validation": packet_validation,
        "final_assessor_execution": final_run_report,
    }
    _write_json(metrics_path, metrics_payload)

    outputs.extend(
        [
            repo_root / POST_ASSESSMENTS_PATH,
            repo_root / PUBLIC_POST_ASSESSMENTS_PATH,
            corrective_results_path,
            public_corrective_results_path,
            evidence_ledgers_path,
            generation_packets_path,
            metrics_path,
            public_report_path,
            repo_root / (PHASE9B_ROOT / "post_correction_encoding_normalization_report.json"),
        ]
    )
    manifest, manifest_path = _write_complete_manifest(
        repo_root,
        outputs,
        metrics_payload,
        final_run_report,
        normalization_report,
        final_normalization_report,
    )
    return Phase9BResult(
        status="complete",
        initial_packets=packets,
        public_initial_packets=public_packets,
        outputs=outputs,
        manifest=manifest,
        manifest_path=manifest_path,
        missing_assessment_files=[],
    )


__all__ = [
    "ASSESSMENT_CONTEXT_K",
    "CORRECTION_STRATEGIES",
    "EVIDENCE_STATUSES",
    "FINAL_CONTEXT_K",
    "CorrectionDecision",
    "EvidenceAssessment",
    "EvidenceAssessmentImportError",
    "EvidenceLedger",
    "EvidenceRequirement",
    "FinalEvidenceAssessment",
    "InitialEvidenceAssessment",
    "Phase9BResult",
    "RetrievalTool",
    "TraceEvent",
    "assessment_to_ledger",
    "build_combined_evidence_pool",
    "build_initial_assessment_packets",
    "build_post_correction_packet",
    "make_trace_event",
    "normalize_initial_assessment_inputs",
    "retrieval_tool_registry",
    "run_phase9b",
    "validate_generator_packets_safe",
    "validate_initial_assessment_packets",
    "validate_initial_assessments",
    "validate_post_correction_assessments",
    "validate_runtime_trace",
    "write_post_correction_single_prompts",
]
